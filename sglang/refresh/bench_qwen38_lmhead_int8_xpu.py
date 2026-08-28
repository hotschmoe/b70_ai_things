#!/usr/bin/env python3
"""Gate selective Qwen3.8 LM-head RTN INT8 on one XPU.

The full checkpoint stores an untied BF16 vocabulary head. TP=2 shards its
output rows, so this oracle benchmarks one real rank-local half. It is a
microbenchmark and numerical gate only; end-to-end greedy equality remains a
separate required serve gate.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--row-chunk", type=int, default=4096)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def load_tp2_lm_head(model: Path, tp_rank: int) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    key = "lm_head.weight"
    shard = model / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise RuntimeError(
            f"expected a 2D BF16 LM head, got {weight.dtype} {tuple(weight.shape)}"
        )
    if weight.shape[0] % 2:
        raise RuntimeError(f"LM-head vocabulary rows are not TP2 divisible: {weight.shape}")
    rows = weight.shape[0] // 2
    return weight[tp_rank * rows : (tp_rank + 1) * rows].contiguous()


def rtn_int8_chunked(
    weight: torch.Tensor, row_chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = torch.empty(weight.shape, dtype=torch.int8)
    scale = torch.empty((weight.shape[0],), dtype=torch.float32)
    for start in range(0, weight.shape[0], row_chunk):
        end = min(start + row_chunk, weight.shape[0])
        rows = weight[start:end].float()
        row_scale = rows.abs().amax(dim=1).clamp_min(1.0e-10) / 127.0
        quant[start:end] = torch.round(rows / row_scale.reshape(-1, 1)).clamp(
            -127, 127
        ).to(torch.int8)
        scale[start:end] = row_scale
    return quant, scale


def accuracy(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    delta = actual_f - reference_f
    topk = min(32, actual_f.numel())
    actual_top = set(torch.topk(actual_f, topk).indices.tolist())
    reference_top = set(torch.topk(reference_f, topk).indices.tolist())
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual_f, reference_f, dim=0)
        ),
        "rel_l2": float(delta.norm() / reference_f.norm().clamp_min(1.0e-12)),
        "max_abs": float(delta.abs().max()),
        "finite": bool(torch.isfinite(actual_f).all()),
        "argmax_equal": int(actual_f.argmax()) == int(reference_f.argmax()),
        "top32_overlap": len(actual_top & reference_top),
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


def main() -> None:
    args = parse_args()
    host_weight = load_tp2_lm_head(args.model, args.tp_rank)
    host_quant, host_scale = rtn_int8_chunked(host_weight, args.row_chunk)
    n, k = host_weight.shape
    generator = torch.Generator().manual_seed(20260828 + n + k)
    x = torch.randn((1, k), generator=generator).to(torch.bfloat16).to("xpu")
    weight = host_weight.to("xpu")
    weight_nk = host_quant.to("xpu")
    weight_nt = weight_nk.t()
    scale = host_scale.to("xpu")
    if weight_nt.stride(0) != 1:
        raise RuntimeError(f"LM-head INT8 weight is not NT-strided: {weight_nt.stride()}")

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
        or not result_accuracy["argmax_equal"]
    ):
        raise RuntimeError(f"LM-head RTN INT8 accuracy failed: {result_accuracy}")

    candidate_host = candidate.cpu()
    repeated = [int8().cpu() for _ in range(16)]
    repeat_exact = all(torch.equal(item, candidate_host) for item in repeated)
    if not repeat_exact:
        raise RuntimeError("LM-head RTN INT8 output is not deterministic")

    result = {
        "torch": torch.__version__,
        "extension": vllm_xpu_kernels._xpu_C.__file__,
        "device": torch.xpu.get_device_name(0),
        "model": str(args.model),
        "method": "TP2 output-shard per-output-channel RTN INT8",
        "tp_rank": args.tp_rank,
        "shape": {"m": 1, "k": k, "n": n},
        "weight_stride": list(weight_nt.stride()),
        "accuracy": result_accuracy,
        "repeat_exact": repeat_exact,
        "bf16_eager": measure(bf16, args.warmup, args.iterations, args.batches),
        "int8_eager": measure(int8, args.warmup, args.iterations, args.batches),
        "bf16_graph": graph_gate(
            bf16, reference, args.warmup, args.iterations, args.batches
        ),
        "int8_graph": graph_gate(
            int8, candidate, args.warmup, args.iterations, args.batches
        ),
    }
    bf16_ms = result["bf16_graph"]["timing"]["median_ms"]
    int8_ms = result["int8_graph"]["timing"]["median_ms"]
    result["graph_saved_ms_per_token_per_rank"] = bf16_ms - int8_ms
    result["graph_speedup"] = bf16_ms / int8_ms
    result["speed_gate_pass"] = int8_ms < bf16_ms
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    del weight, weight_nk, weight_nt, scale, reference, candidate
    gc.collect()
    torch.xpu.empty_cache()
    if not result["speed_gate_pass"]:
        raise RuntimeError(
            "LM-head RTN INT8 did not improve graph time: "
            f"bf16={bf16_ms:.6f} ms int8={int8_ms:.6f} ms"
        )


if __name__ == "__main__":
    main()
