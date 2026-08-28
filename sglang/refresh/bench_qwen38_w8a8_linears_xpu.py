#!/usr/bin/env python3
"""Benchmark real Qwen3.8 TP=2 W8A8 decode projections on one XPU."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import vllm_xpu_kernels._xpu_C  # noqa: F401


CASES = {
    "gate_up": (
        ("model.language_model.layers.0.mlp.gate_proj", "out"),
        ("model.language_model.layers.0.mlp.up_proj", "out"),
    ),
    "down": (("model.language_model.layers.0.mlp.down_proj", "in"),),
    "qkv": (
        ("model.language_model.layers.3.self_attn.q_proj", "out"),
        ("model.language_model.layers.3.self_attn.k_proj", "out"),
        ("model.language_model.layers.3.self_attn.v_proj", "out"),
    ),
    "out": (("model.language_model.layers.3.self_attn.o_proj", "in"),),
}

CALLS_PER_TOKEN = {"gate_up": 64, "down": 64, "qkv": 16, "out": 16}


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


def tp2_slice(
    weight: torch.Tensor, scale: torch.Tensor, axis: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if axis == "out":
        rows = weight.shape[0] // 2
        return weight[:rows].contiguous(), scale[:rows].contiguous()
    if axis == "in":
        columns = weight.shape[1] // 2
        return weight[:, :columns].contiguous(), scale.contiguous()
    raise ValueError(axis)


def load_case(model: Path, parts: tuple[tuple[str, str], ...]):
    weights = []
    scales = []
    for base, axis in parts:
        weight = load_tensor(model, f"{base}.weight")
        scale = load_tensor(model, f"{base}.weight_scale").reshape(-1)
        if weight.dtype != torch.int8:
            raise RuntimeError(f"{base}: expected INT8 weight, got {weight.dtype}")
        weight, scale = tp2_slice(weight, scale, axis)
        weights.append(weight)
        scales.append(scale)
    return torch.cat(weights, dim=0), torch.cat(scales).to(torch.float32)


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    cosine = float(
        torch.nn.functional.cosine_similarity(actual_f, reference_f, dim=0)
    )
    rel_l2 = float(delta.norm() / reference_f.norm().clamp_min(1.0e-12))
    return {
        "cosine": cosine,
        "rel_l2": rel_l2,
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

    exact = []
    eager_host = eager.cpu()
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
    parts: tuple[tuple[str, str], ...],
    model: Path,
    warmup: int,
    iterations: int,
    batches: int,
) -> dict[str, Any]:
    host_weight_nk, host_scale = load_case(model, parts)
    n, k = host_weight_nk.shape
    generator = torch.Generator().manual_seed(20260828 + n + k)
    x = torch.randn((1, k), generator=generator).to(torch.bfloat16).to("xpu")
    weight_nk = host_weight_nk.to("xpu")
    weight_nt = weight_nk.t()
    weight_scale = host_scale.to("xpu")
    if weight_nt.stride(0) != 1:
        raise RuntimeError(f"{name}: weight is not the required NT view")

    dequant_weight = weight_nk.to(torch.bfloat16).mul(
        weight_scale.to(torch.bfloat16).reshape(-1, 1)
    )
    reference = torch.nn.functional.linear(x, dequant_weight)
    torch.xpu.synchronize()
    del dequant_weight

    def quantize():
        return torch.ops._xpu_C.per_token_quant_int8_xpu(x)

    def current_full():
        quant, scale = quantize()
        return torch.ops._xpu_C.int8_gemm_w8a8(
            quant,
            scale,
            weight_nt,
            weight_scale,
            torch.bfloat16,
            None,
        )

    def w8a16():
        return torch.ops._xpu_C.int8_gemm_w8a16(
            x, weight_nt, weight_scale, None
        )

    current = current_full()
    candidate = w8a16()
    torch.xpu.synchronize()

    repeated = []
    for _ in range(16):
        repeated.append(w8a16().cpu())
    candidate_host = candidate.cpu()
    deterministic = all(torch.equal(item, candidate_host) for item in repeated)
    if not deterministic:
        raise RuntimeError(f"{name}: W8A16 eager output is not deterministic")

    current_accuracy = metrics(current, reference)
    candidate_accuracy = metrics(candidate, reference)
    if (
        not current_accuracy["finite"]
        or current_accuracy["cosine"] < 0.999
        or current_accuracy["rel_l2"] > 0.02
    ):
        raise RuntimeError(f"{name}: current W8A8 accuracy failed: {current_accuracy}")
    if (
        not candidate_accuracy["finite"]
        or candidate_accuracy["cosine"] < 0.999
        or candidate_accuracy["rel_l2"] > 0.01
    ):
        raise RuntimeError(
            f"{name}: candidate W8A16 accuracy failed: {candidate_accuracy}"
        )

    result = {
        "shape": {"m": 1, "k": k, "n": n},
        "weight_stride": list(weight_nt.stride()),
        "current_accuracy": current_accuracy,
        "w8a16_accuracy": candidate_accuracy,
        "w8a16_repeat_exact": deterministic,
        "quantize": measure(quantize, warmup, iterations, batches),
        "current_eager": measure(current_full, warmup, iterations, batches),
        "w8a16_eager": measure(w8a16, warmup, iterations, batches),
        "current_graph": graph_gate(
            current_full, current, warmup, iterations, batches
        ),
        "w8a16_graph": graph_gate(
            w8a16, candidate, warmup, iterations, batches
        ),
    }
    del x, weight_nk, weight_nt, weight_scale, reference, current, candidate
    gc.collect()
    torch.xpu.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    cases = {
        name: run_case(
            name,
            parts,
            args.model,
            args.warmup,
            args.iterations,
            args.batches,
        )
        for name, parts in CASES.items()
    }
    weighted_current = sum(
        CALLS_PER_TOKEN[name]
        * cases[name]["current_graph"]["timing"]["median_ms"]
        for name in CASES
    )
    weighted_w8a16 = sum(
        CALLS_PER_TOKEN[name]
        * cases[name]["w8a16_graph"]["timing"]["median_ms"]
        for name in CASES
    )
    result = {
        "torch": torch.__version__,
        "extension": vllm_xpu_kernels._xpu_C.__file__,
        "device": torch.xpu.get_device_name(0),
        "model": str(args.model),
        "calls_per_token_per_rank": CALLS_PER_TOKEN,
        "cases": cases,
        "weighted_graph_ms_per_token_per_rank": {
            "current": weighted_current,
            "w8a16": weighted_w8a16,
            "saved": weighted_current - weighted_w8a16,
            "speedup": weighted_current / weighted_w8a16,
        },
    }
    speed_gate_pass = weighted_w8a16 < weighted_current
    result["speed_gate_pass"] = speed_gate_pass
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    if not speed_gate_pass:
        raise RuntimeError(
            "W8A16 did not improve the weighted real-shape graph projection time: "
            f"{result['weighted_graph_ms_per_token_per_rank']}"
        )


if __name__ == "__main__":
    main()
