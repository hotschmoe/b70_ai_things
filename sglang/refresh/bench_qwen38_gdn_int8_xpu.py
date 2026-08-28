#!/usr/bin/env python3
"""Gate selective Qwen3.8 GDN RTN INT8 projections on one XPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batches", type=int, default=7)
    return parser.parse_args()


def load_tensor(model: Path, key: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def load_cases(model: Path) -> dict[str, torch.Tensor]:
    config = json.loads((model / "config.json").read_text())
    config = config.get("text_config", config)
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    qkv = load_tensor(
        model, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    z = load_tensor(
        model, "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    )
    q, k, v = qkv.split((key_dim, key_dim, value_dim), dim=0)
    qkvz = torch.cat(
        (
            q[: key_dim // 2],
            k[: key_dim // 2],
            v[: value_dim // 2],
            z[: value_dim // 2],
        ),
        dim=0,
    ).contiguous()
    out = load_tensor(
        model, "model.language_model.layers.0.linear_attn.out_proj.weight"
    )[:, : value_dim // 2].contiguous()
    return {"qkvz": qkvz, "out": out}


def rtn_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_float = weight.float()
    scale = weight_float.abs().amax(dim=1).clamp_min(1.0e-10) / 127.0
    quant = torch.round(weight_float / scale.reshape(-1, 1))
    return quant.clamp(-127, 127).to(torch.int8), scale


def accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual_f, reference_f, dim=0)
        ),
        "rel_l2": float(delta.norm() / reference_f.norm().clamp_min(1.0e-12)),
        "max_abs": float(delta.abs().max()),
        "finite": bool(torch.isfinite(actual_f).all()),
    }


def measure(
    call: Callable[[], Any], warmup: int, iterations: int, batches: int
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.xpu.synchronize()
    samples = []
    for _ in range(batches):
        torch.xpu.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / iterations)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cv": statistics.pstdev(samples) / statistics.mean(samples),
    }


def graph_gate(
    call: Callable[[], torch.Tensor],
    eager: torch.Tensor,
    warmup: int,
    iterations: int,
    batches: int,
) -> dict[str, Any]:
    for _ in range(4):
        call()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        static_output = call()
    torch.xpu.synchronize()
    eager_host = eager.cpu()
    exact = []
    for _ in range(16):
        graph.replay()
        torch.xpu.synchronize()
        exact.append(torch.equal(static_output.cpu(), eager_host))
    if not all(exact):
        raise RuntimeError(f"XPUGraph replay mismatch: {exact}")
    return {
        "replays": len(exact),
        "replay_exact": True,
        "timing": measure(graph.replay, warmup, iterations, batches),
    }


def run_case(
    name: str,
    host_weight: torch.Tensor,
    warmup: int,
    iterations: int,
    batches: int,
) -> dict[str, Any]:
    host_quant, host_scale = rtn_int8(host_weight)
    n, k = host_weight.shape
    generator = torch.Generator().manual_seed(20260828 + n + k)
    x = torch.randn((1, k), generator=generator).to(torch.bfloat16).to("xpu")
    weight = host_weight.to("xpu")
    weight_nk = host_quant.to("xpu")
    weight_nt = weight_nk.t()
    scale = host_scale.to(torch.float32).to("xpu")

    def bf16():
        return torch.nn.functional.linear(x, weight)

    def int8():
        quant, input_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
        return torch.ops._xpu_C.int8_gemm_w8a8(
            quant,
            input_scale,
            weight_nt,
            scale,
            torch.bfloat16,
            None,
        )

    reference = bf16()
    candidate = int8()
    torch.xpu.synchronize()
    result_accuracy = accuracy(candidate, reference)
    if (
        not result_accuracy["finite"]
        or result_accuracy["cosine"] < 0.999
        or result_accuracy["rel_l2"] > 0.02
    ):
        raise RuntimeError(f"{name}: RTN INT8 accuracy failed: {result_accuracy}")

    repeated = []
    for _ in range(16):
        repeated.append(int8().cpu())
    candidate_host = candidate.cpu()
    deterministic = all(torch.equal(item, candidate_host) for item in repeated)
    if not deterministic:
        raise RuntimeError(f"{name}: RTN INT8 output is not deterministic")

    return {
        "shape": {"m": 1, "k": k, "n": n},
        "weight_stride": list(weight_nt.stride()),
        "accuracy": result_accuracy,
        "repeat_exact": deterministic,
        "bf16_eager": measure(bf16, warmup, iterations, batches),
        "int8_eager": measure(int8, warmup, iterations, batches),
        "bf16_graph": graph_gate(bf16, reference, warmup, iterations, batches),
        "int8_graph": graph_gate(int8, candidate, warmup, iterations, batches),
    }


def main() -> None:
    args = parse_args()
    cases = {
        name: run_case(
            name, weight, args.warmup, args.iterations, args.batches
        )
        for name, weight in load_cases(args.model).items()
    }
    calls_per_token = {"qkvz": 48, "out": 48}
    weighted_bf16 = sum(
        calls_per_token[name]
        * cases[name]["bf16_graph"]["timing"]["median_ms"]
        for name in cases
    )
    weighted_int8 = sum(
        calls_per_token[name]
        * cases[name]["int8_graph"]["timing"]["median_ms"]
        for name in cases
    )
    speed_gate_pass = weighted_int8 < weighted_bf16
    result = {
        "torch": torch.__version__,
        "extension": vllm_xpu_kernels._xpu_C.__file__,
        "device": torch.xpu.get_device_name(0),
        "model": str(args.model),
        "method": "symmetric per-output-channel RTN INT8",
        "calls_per_token_per_rank": calls_per_token,
        "cases": cases,
        "weighted_graph_ms_per_token_per_rank": {
            "bf16": weighted_bf16,
            "int8": weighted_int8,
            "saved": weighted_bf16 - weighted_int8,
            "speedup": weighted_bf16 / weighted_int8,
        },
        "speed_gate_pass": speed_gate_pass,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    if not speed_gate_pass:
        raise RuntimeError(
            "GDN RTN INT8 did not improve weighted real-shape graph time: "
            f"{result['weighted_graph_ms_per_token_per_rank']}"
        )


if __name__ == "__main__":
    main()
