#!/usr/bin/env python3
"""Benchmark real Qwen3.8 TP=2 decode linear shapes on one Intel XPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401


FP8_CASES = {
    "gdn_qkvz": (
        ("model.language_model.layers.0.linear_attn.in_proj_qkv", "out"),
        ("model.language_model.layers.0.linear_attn.in_proj_z", "out"),
    ),
    "full_attn_qkv": (
        ("model.language_model.layers.3.self_attn.q_proj", "out"),
        ("model.language_model.layers.3.self_attn.k_proj", "out"),
        ("model.language_model.layers.3.self_attn.v_proj", "out"),
    ),
    "common_out": (
        ("model.language_model.layers.0.linear_attn.out_proj", "in"),
    ),
}

NVFP4_CASES = {
    "mlp_gate": ("model.language_model.layers.0.mlp.gate_proj", "out"),
    "mlp_down": ("model.language_model.layers.0.mlp.down_proj", "in"),
    "lm_head": ("lm_head", "out"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--m1-so", type=Path)
    parser.add_argument("--family", choices=("fp8", "nvfp4", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def load_tensor(model: Path, key: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def timed(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def compare(name: str, actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    cosine = float(torch.nn.functional.cosine_similarity(actual_f, reference_f, dim=0))
    rel_l2 = float(delta.norm() / reference_f.norm().clamp_min(1e-12))
    print(f"  {name}: cosine={cosine:.8f} rel_l2={rel_l2:.8g}")
    if not torch.isfinite(actual_f).all() or cosine < 0.999 or rel_l2 > 0.01:
        raise RuntimeError(f"{name}: failed agreement gate")


def tp2_slice(weight: torch.Tensor, scale: torch.Tensor | None, axis: str):
    if axis == "out":
        midpoint = weight.shape[0] // 2
        weight = weight[:midpoint].contiguous()
        if scale is not None:
            scale = scale[:midpoint].contiguous()
    elif axis == "in":
        if weight.dtype == torch.uint8:
            midpoint = weight.shape[1] // 2
            weight = weight[:, :midpoint].contiguous()
            if scale is not None:
                scale = scale[:, : scale.shape[1] // 2].contiguous()
        else:
            midpoint = weight.shape[1] // 2
            weight = weight[:, :midpoint].contiguous()
    else:
        raise ValueError(axis)
    return weight, scale


def load_fused_fp8(model: Path, parts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loaded = []
    weight_scales = []
    input_scales = []
    for base, axis in parts:
        weight = load_tensor(model, f"{base}.weight")
        weight, _ = tp2_slice(weight, None, axis)
        loaded.append(weight.to("xpu"))
        weight_scales.append(
            load_tensor(model, f"{base}.weight_scale").max().to(torch.float32)
        )
        input_scales.append(
            load_tensor(model, f"{base}.input_scale").max().to(torch.float32)
        )

    input_scale = torch.stack(input_scales).max()
    if not torch.allclose(torch.stack(input_scales), input_scale):
        raise RuntimeError("fused FP8 input scales differ")
    weight_scale = torch.stack(weight_scales).max().to(device="xpu").reshape(1)
    fp8_limit = torch.finfo(torch.float8_e4m3fn).max
    requantized = []
    for weight, scale in zip(loaded, weight_scales):
        requantized.append(
            weight.float()
            .mul(scale.to(device="xpu"))
            .div(weight_scale)
            .clamp(-fp8_limit, fp8_limit)
            .to(torch.float8_e4m3fn)
        )
    return (
        torch.cat(requantized, dim=0).t(),
        weight_scale,
        input_scale.to(device="xpu").reshape(1),
    )


def graph_replay_check(
    name: str,
    fn: Callable[[], torch.Tensor],
    x: torch.Tensor,
) -> None:
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    graph = torch.xpu.XPUGraph()
    with torch.xpu.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()
        with torch.xpu.graph(graph):
            graph_output = fn()
    torch.xpu.current_stream().wait_stream(stream)
    torch.xpu.synchronize()

    replacement = torch.randn_like(x)
    x.copy_(replacement)
    eager = fn()
    graph.replay()
    torch.xpu.synchronize()
    compare(f"{name} graph replay vs eager", graph_output, eager)


def bench_fp8(args: argparse.Namespace) -> None:
    fp8_limit = torch.finfo(torch.float8_e4m3fn).max
    for case_index, (label, parts) in enumerate(FP8_CASES.items()):
        weight_t, weight_scale, input_scale = load_fused_fp8(args.model, parts)
        k, n = weight_t.shape
        torch.manual_seed(20260828)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="xpu")

        def quantize() -> torch.Tensor:
            return x.float().div(input_scale).clamp(-fp8_limit, fp8_limit).to(
                torch.float8_e4m3fn
            )

        qinput = quantize()

        def stock_gemm() -> torch.Tensor:
            output = torch._scaled_mm(
                qinput,
                weight_t,
                scale_a=input_scale,
                scale_b=weight_scale,
                out_dtype=torch.bfloat16,
            )
            return output[0] if isinstance(output, tuple) else output

        def direct_gemm() -> torch.Tensor:
            return torch.ops._xpu_C.fp8_gemm(
                qinput,
                weight_t,
                torch.bfloat16,
                input_scale,
                weight_scale,
                None,
            )

        def w8a16_gemm() -> torch.Tensor:
            return torch.ops._xpu_C.fp8_gemm_w8a16(
                x, weight_t, weight_scale, None
            )

        stock = stock_gemm()
        direct = direct_gemm()
        w8a16 = w8a16_gemm()
        a16_reference = torch.nn.functional.linear(
            x,
            weight_t.t().float().mul(weight_scale).to(torch.bfloat16),
        )
        torch.xpu.synchronize()
        print(f"FP8 {label}: M=1 N={n} K={k}")
        compare("direct-w8a8 vs stock", direct, stock)
        compare("direct-w8a16 vs dequant reference", w8a16, a16_reference)

        timings = {
            "quant": timed(quantize, args.warmup, args.iterations),
            "stock_gemm": timed(stock_gemm, args.warmup, args.iterations),
            "direct_gemm": timed(direct_gemm, args.warmup, args.iterations),
            "w8a16_gemm": timed(w8a16_gemm, args.warmup, args.iterations),
            "quant_stock": timed(
                lambda: torch._scaled_mm(
                    quantize(),
                    weight_t,
                    scale_a=input_scale,
                    scale_b=weight_scale,
                    out_dtype=torch.bfloat16,
                ),
                args.warmup,
                args.iterations,
            ),
            "quant_direct": timed(
                lambda: torch.ops._xpu_C.fp8_gemm(
                    quantize(),
                    weight_t,
                    torch.bfloat16,
                    input_scale,
                    weight_scale,
                    None,
                ),
                args.warmup,
                args.iterations,
            ),
        }
        print("  " + " ".join(f"{name}={value:.4f}ms" for name, value in timings.items()))
        if case_index == 0:
            graph_replay_check("direct-w8a16", w8a16_gemm, x)
        del weight_t, x, qinput, stock, direct, w8a16, a16_reference
        torch.xpu.empty_cache()


def bench_nvfp4(args: argparse.Namespace) -> None:
    if args.m1_so is None:
        raise ValueError("--m1-so is required for the NVFP4 benchmark")
    torch.ops.load_library(str(args.m1_so))
    if not hasattr(torch.ops.b70_nvfp4_m1, "gemv"):
        raise RuntimeError("M=1 sidecar did not register b70_nvfp4_m1.gemv")

    for label, (base, axis) in NVFP4_CASES.items():
        weight = load_tensor(args.model, f"{base}.weight")
        block_scale = load_tensor(args.model, f"{base}.weight_scale")
        global_scale = load_tensor(args.model, f"{base}.weight_scale_2")
        weight, block_scale = tp2_slice(weight, block_scale, axis)
        folded_scale = block_scale.float().mul(global_scale.float()).to(torch.bfloat16)

        weight = weight.to("xpu")
        block_scale_nt = block_scale.to("xpu").t().contiguous()
        folded_scale_nt = folded_scale.to("xpu").t().contiguous()
        global_scale = global_scale.to(device="xpu", dtype=torch.float32).reshape(1)
        n, packed_k = weight.shape
        k = packed_k * 2
        torch.manual_seed(20260828)
        x = torch.randn(1, k, dtype=torch.bfloat16, device="xpu")

        def onednn() -> torch.Tensor:
            return torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
                x, weight.t(), None, block_scale_nt, global_scale, 16
            )

        def esimd() -> torch.Tensor:
            return torch.ops.b70_nvfp4_m1.gemv(x, weight, folded_scale_nt)

        reference = onednn()
        candidate = esimd()
        repeat = esimd()
        torch.xpu.synchronize()
        print(f"NVFP4 {label}: M=1 N={n} K={k}")
        compare("esimd vs onednn", candidate, reference)
        if not torch.equal(candidate, repeat):
            raise RuntimeError(f"{label}: ESIMD output is not deterministic")
        one_ms = timed(onednn, args.warmup, args.iterations)
        esimd_ms = timed(esimd, args.warmup, args.iterations)
        print(
            f"  onednn={one_ms:.4f}ms esimd={esimd_ms:.4f}ms "
            f"speedup={one_ms / esimd_ms:.3f}x"
        )
        del weight, block_scale_nt, folded_scale_nt, global_scale, x
        del reference, candidate, repeat
        torch.xpu.empty_cache()


def main() -> None:
    args = parse_args()
    print(f"torch={torch.__version__} device={torch.xpu.get_device_name(0)}")
    if args.family in ("fp8", "all"):
        bench_fp8(args)
    if args.family in ("nvfp4", "all"):
        bench_nvfp4(args)
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
