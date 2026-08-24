#!/usr/bin/env python3
"""Exact-M=11 C4 W8A16 routing microbenchmark for Intel XPU.

This compares the real SGLang paths on per-rank Qwen3.6-27B shapes:

  BF16:       BF16 activation x BF16 weight
  current:    BF16 -> FP16 -> dynamic INT8 quant -> W8A8 -> BF16
  candidate:  BF16 -> FP16 -> W8A16 -> BF16

The current and candidate paths share one [K,N] stride-0-1 INT8 weight view,
matching sglang/patches/w8a8_shim.py. No graph capture or queue profiling is
used. Two position-balanced repeat blocks report XPU-event and synchronized
wall time. The result is written before a failing gate exits nonzero.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


# Imported only by main so CPU-only contract tests do not require a host torch
# installation. GPU helpers use this module global after main initializes it.
torch = None


M = 11
SHAPES = (
    # name, K, N, calls per target decode step in the combined trace contract
    ("gdn_qkvz", 5120, 8192, 48),
    ("gdn_and_attn_out", 3072, 5120, 64),
    ("mlp_gate_up", 5120, 17408, 64),
    ("mlp_down", 8704, 5120, 64),
    ("attn_qkv", 5120, 7168, 16),
)
REQUIRED_OPS = (
    "dynamic_per_token_int8_quant",
    "int8_gemm_w8a8",
    "int8_gemm_w8a16",
)


def cv_pct(values):
    mean = statistics.fmean(values)
    return 100.0 * statistics.pstdev(values) / mean if mean else math.inf


def relative_l2(actual, expected):
    delta = actual.float() - expected.float()
    denominator = expected.float().norm().item()
    return delta.norm().item() / max(denominator, 1e-12)


def quantize_weight(weight_bf16, chunk_rows=1024):
    rows, cols = weight_bf16.shape
    quant = torch.empty((rows, cols), dtype=torch.int8, device=weight_bf16.device)
    scale = torch.empty(rows, dtype=torch.float16, device=weight_bf16.device)
    for row0 in range(0, rows, chunk_rows):
        row1 = min(row0 + chunk_rows, rows)
        values = weight_bf16[row0:row1].float()
        scales = values.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8) / 127.0
        quant[row0:row1] = torch.round(values / scales).clamp_(-127, 127).to(
            torch.int8
        )
        scale[row0:row1] = scales.reshape(-1).to(torch.float16)
    return quant, scale


def time_block(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        fn()
    end_event.record()
    torch.xpu.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iterations
    device_ms = start_event.elapsed_time(end_event) / iterations
    return {"device_us": device_ms * 1000.0, "wall_us": wall_ms * 1000.0}


def make_paths(ops, x_bf16, weight_bf16, weight_nt, weight_scale):
    def bf16_path():
        return torch.matmul(x_bf16, weight_bf16.t())

    def current_path():
        xf = x_bf16.to(torch.float16).contiguous()
        xq, xs, _xz = ops.dynamic_per_token_int8_quant(xf, True, 8)
        output = ops.int8_gemm_w8a8(
            xq,
            xs.contiguous(),
            None,
            weight_nt,
            weight_scale,
            None,
            None,
            torch.float16,
        )
        return output.to(torch.bfloat16)

    def w8a16_path():
        xf = x_bf16.to(torch.float16).contiguous()
        output = ops.int8_gemm_w8a16(xf, weight_nt, weight_scale, None)
        return output.to(torch.bfloat16)

    return {"bf16": bf16_path, "current_w8a8": current_path, "w8a16": w8a16_path}


def benchmark_shape(ops, shape, warmup, iterations, repeats, seed):
    name, k, n, calls = shape
    torch.manual_seed(seed)
    x_bf16 = (torch.randn((M, k), device="xpu", dtype=torch.bfloat16) * 0.05)
    weight_bf16 = (
        torch.randn((n, k), device="xpu", dtype=torch.bfloat16) * 0.02
    )
    weight_nk, weight_scale = quantize_weight(weight_bf16)
    weight_nt = weight_nk.t()
    if weight_nt.stride(0) != 1 or tuple(weight_nt.shape) != (k, n):
        raise RuntimeError(
            f"bad shared B_nt contract for {name}: "
            f"shape={tuple(weight_nt.shape)} stride={tuple(weight_nt.stride())}"
        )
    paths = make_paths(ops, x_bf16, weight_bf16, weight_nt, weight_scale)

    with torch.no_grad():
        reference = paths["bf16"]()
        current = paths["current_w8a8"]()
        candidate = paths["w8a16"]()
        torch.xpu.synchronize()
        numerical = {
            "current_w8a8": {
                "relative_l2": relative_l2(current, reference),
                "max_abs": (current.float() - reference.float()).abs().max().item(),
                "finite": bool(torch.isfinite(current).all().item()),
            },
            "w8a16": {
                "relative_l2": relative_l2(candidate, reference),
                "max_abs": (candidate.float() - reference.float()).abs().max().item(),
                "finite": bool(torch.isfinite(candidate).all().item()),
            },
        }

        timings = {path: [] for path in paths}
        forward_order = ("bf16", "current_w8a8", "w8a16")
        reverse_order = tuple(reversed(forward_order))
        for repeat in range(repeats):
            order = forward_order if repeat % 2 == 0 else reverse_order
            block = {}
            for path in order:
                block[path] = time_block(paths[path], warmup, iterations)
            for path in paths:
                timings[path].append(block[path])

    summary = {}
    for path, blocks in timings.items():
        device = [block["device_us"] for block in blocks]
        wall = [block["wall_us"] for block in blocks]
        summary[path] = {
            "repeat_blocks": blocks,
            "device_us_mean": statistics.fmean(device),
            "device_cv_pct": cv_pct(device),
            "wall_us_mean": statistics.fmean(wall),
            "wall_cv_pct": cv_pct(wall),
        }
    current_us = summary["current_w8a8"]["device_us_mean"]
    candidate_us = summary["w8a16"]["device_us_mean"]
    gain_pct = 100.0 * (current_us - candidate_us) / current_us
    result = {
        "name": name,
        "m": M,
        "k": k,
        "n": n,
        "calls_per_step": calls,
        "weight_nt_stride": list(weight_nt.stride()),
        "numerical": numerical,
        "timings": summary,
        "w8a16_gain_pct": gain_pct,
    }
    del paths, reference, current, candidate
    del x_bf16, weight_bf16, weight_nk, weight_nt, weight_scale
    torch.xpu.empty_cache()
    return result


def analyze(results, minimum_gain_pct, maximum_cv_pct, maximum_relative_l2):
    by_name = {item["name"]: item for item in results}
    required_gate_shapes = ("gdn_qkvz", "gdn_and_attn_out")
    weighted_current = sum(
        item["timings"]["current_w8a8"]["device_us_mean"]
        * item["calls_per_step"]
        for item in results
    )
    weighted_candidate = sum(
        item["timings"]["w8a16"]["device_us_mean"] * item["calls_per_step"]
        for item in results
    )
    gate_current = sum(
        by_name[name]["timings"]["current_w8a8"]["device_us_mean"]
        * by_name[name]["calls_per_step"]
        for name in required_gate_shapes
    )
    gate_candidate = sum(
        by_name[name]["timings"]["w8a16"]["device_us_mean"]
        * by_name[name]["calls_per_step"]
        for name in required_gate_shapes
    )
    checks = {
        "exact_shape_set": set(by_name)
        == {
            "gdn_qkvz",
            "gdn_and_attn_out",
            "mlp_gate_up",
            "mlp_down",
            "attn_qkv",
        },
        "qkvz_gain_ge_threshold": by_name["gdn_qkvz"]["w8a16_gain_pct"]
        >= minimum_gain_pct,
        "out_gain_ge_threshold": by_name["gdn_and_attn_out"]["w8a16_gain_pct"]
        >= minimum_gain_pct,
        "qkvz_out_weighted_gain_ge_threshold": (
            100.0 * (gate_current - gate_candidate) / gate_current
        )
        >= minimum_gain_pct,
        "all_candidate_shape_nonregression": all(
            item["w8a16_gain_pct"] >= 0.0 for item in results
        ),
        "all_current_and_candidate_cv_bounded": all(
            item["timings"][path]["device_cv_pct"] <= maximum_cv_pct
            for item in results
            for path in ("current_w8a8", "w8a16")
        ),
        "all_outputs_finite": all(
            item["numerical"][path]["finite"]
            for item in results
            for path in ("current_w8a8", "w8a16")
        ),
        "all_relative_l2_bounded": all(
            item["numerical"][path]["relative_l2"] <= maximum_relative_l2
            for item in results
            for path in ("current_w8a8", "w8a16")
        ),
        "w8a16_not_less_accurate_than_w8a8": all(
            item["numerical"]["w8a16"]["relative_l2"]
            <= item["numerical"]["current_w8a8"]["relative_l2"] + 1e-4
            for item in results
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "qkvz_out_weighted": {
            "current_us_per_step": gate_current,
            "w8a16_us_per_step": gate_candidate,
            "gain_pct": 100.0 * (gate_current - gate_candidate) / gate_current,
        },
        "all_routes_weighted": {
            "current_us_per_step": weighted_current,
            "w8a16_us_per_step": weighted_candidate,
            "gain_pct": 100.0
            * (weighted_current - weighted_candidate)
            / weighted_current,
        },
    }


def main():
    global torch
    import torch as torch_module

    torch = torch_module
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", type=Path, default=Path("/work/kernel/_xpu_C.abi3.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--minimum-gain-pct", type=float, default=5.0)
    parser.add_argument("--maximum-cv-pct", type=float, default=5.0)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.10)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 10 or args.repeats < 2:
        raise SystemExit("requires warmup>=1, iterations>=10, repeats>=2")
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise SystemExit("requires exactly one visible XPU")
    ctypes.CDLL(str(args.so), mode=ctypes.RTLD_GLOBAL)
    ops = torch.ops._xpu_C
    missing = [name for name in REQUIRED_OPS if not hasattr(ops, name)]
    if missing:
        raise SystemExit(f"missing required kernel ops: {missing}")

    results = []
    for index, shape in enumerate(SHAPES):
        print(
            f"BENCH shape={shape[0]} M={M} K={shape[1]} N={shape[2]} "
            f"calls={shape[3]}",
            flush=True,
        )
        result = benchmark_shape(
            ops,
            shape,
            args.warmup,
            args.iterations,
            args.repeats,
            seed=7100 + index,
        )
        results.append(result)
        print(
            f"RESULT shape={shape[0]} "
            f"bf16_us={result['timings']['bf16']['device_us_mean']:.3f} "
            f"current_us={result['timings']['current_w8a8']['device_us_mean']:.3f} "
            f"w8a16_us={result['timings']['w8a16']['device_us_mean']:.3f} "
            f"gain_pct={result['w8a16_gain_pct']:.3f}",
            flush=True,
        )
    analysis = analyze(
        results,
        args.minimum_gain_pct,
        args.maximum_cv_pct,
        args.maximum_relative_l2,
    )
    payload = {
        "schema": "c4_m11_w8a16_microbench_v1",
        "environment": {
            "torch": torch.__version__,
            "xpu_count": torch.xpu.device_count(),
            "so": str(args.so),
            "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
            "ccl_topo_p2p_access": os.environ.get("CCL_TOPO_P2P_ACCESS"),
        },
        "parameters": {
            "m": M,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "minimum_gain_pct": args.minimum_gain_pct,
            "maximum_cv_pct": args.maximum_cv_pct,
            "maximum_relative_l2": args.maximum_relative_l2,
            "timing": "xpu_event_batch_plus_synchronized_wall",
            "position_balance": "forward_then_reverse",
            "queue_profiling": False,
        },
        "shapes": results,
        "analysis": analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    for name, passed in analysis["checks"].items():
        print(f"CHECK {name}={'PASS' if passed else 'FAIL'}", flush=True)
    print(
        f"C4_M11_W8A16 verdict={'PASS' if analysis['pass'] else 'FAIL'} "
        f"qkvz_out_gain_pct={analysis['qkvz_out_weighted']['gain_pct']:.3f} "
        f"all_routes_gain_pct={analysis['all_routes_weighted']['gain_pct']:.3f}",
        flush=True,
    )
    return 0 if analysis["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
