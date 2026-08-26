#!/usr/bin/env python3
"""Summarize bounded e190 XPU Kineto traces without synchronization.

The e190 worker annotates each scheduler iteration as
``execute_context_*``. Kineto joins CPU launch events to device kernels with
the ``correlation`` field. This analyzer reports only events owned by those
iteration ranges, keeping initialization and trace-flush work out of the
ledger.

XPU graph replay is partly opaque to Kineto. Driver-call counts are therefore
reported alongside visible kernels. In the Qwen3.6 PIECEWISE control, fence
resets and queue submissions are the useful structural signature of graph
replay even when captured MoE and collective kernels have no individual
device events.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("traceEvents", payload)


def contains(interval: dict[str, Any], event: dict[str, Any]) -> bool:
    if interval.get("pid") != event.get("pid"):
        return False
    start = float(event.get("ts", -1.0))
    return float(interval["ts"]) <= start <= (
        float(interval["ts"]) + float(interval.get("dur", 0.0))
    )


def kernel_bucket(name: str) -> str:
    lower = name.lower()
    if "oneccl" in lower or "allreduce" in lower or "allgather" in lower:
        return "collective_visible"
    if "gdn::" in lower:
        return "gdn_visible"
    if "decodefwd" in lower or "fmha" in lower:
        return "full_attention_visible"
    if "reshape_and_cache" in lower:
        return "kv_cache_visible"
    if "radixselect" in lower or "radixsort" in lower:
        return "sampler_visible"
    if name == "gemm_kernel":
        return "gemm_visible"
    if "memcpy" in lower:
        return "memcpy_visible"
    return "other_visible"


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(path: Path) -> dict[str, Any]:
    events = [event for event in load_events(path) if event.get("ph") == "X"]
    iterations = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("execute_context_")
    ]
    iterations.sort(key=lambda event: float(event["ts"]))

    runtime_by_correlation: dict[Any, dict[str, Any]] = {}
    for event in events:
        if event.get("cat") != "xpu_runtime":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            runtime_by_correlation[correlation] = event

    per_iteration: list[dict[str, Any]] = []
    kernel_name_us: collections.Counter[str] = collections.Counter()
    kernel_name_count: collections.Counter[str] = collections.Counter()
    bucket_us: collections.Counter[str] = collections.Counter()
    bucket_count: collections.Counter[str] = collections.Counter()

    structural_names = (
        "zeCommandQueueExecuteCommandLists",
        "zeFenceReset",
        "zeEventHostSynchronize",
        "zeCommandListAppendBarrier",
        "zeCommandListAppendLaunchKernel",
        "urEnqueueKernelLaunch",
    )
    structural_counts: dict[str, list[int]] = {
        name: [] for name in structural_names
    }
    structural_us: dict[str, list[float]] = {name: [] for name in structural_names}

    for iteration in iterations:
        owned_cpu = [
            event
            for event in events
            if event.get("cat") in {"cpu_op", "xpu_runtime", "xpu_driver"}
            and event.get("tid") == iteration.get("tid")
            and contains(iteration, event)
        ]
        owned_runtime_correlations = {
            event.get("args", {}).get("correlation")
            for event in owned_cpu
            if event.get("cat") == "xpu_runtime"
        }
        owned_kernels = [
            event
            for event in events
            if event.get("cat") in {"kernel", "gpu_memcpy"}
            and event.get("args", {}).get("correlation")
            in owned_runtime_correlations
        ]

        iteration_buckets: collections.Counter[str] = collections.Counter()
        for kernel in owned_kernels:
            name = str(kernel.get("name", "unknown"))
            duration = float(kernel.get("dur", 0.0))
            bucket = kernel_bucket(name)
            kernel_name_us[name] += duration
            kernel_name_count[name] += 1
            bucket_us[bucket] += duration
            bucket_count[bucket] += 1
            iteration_buckets[bucket] += duration

        structural: dict[str, dict[str, float | int]] = {}
        for name in structural_names:
            matching = [event for event in owned_cpu if event.get("name") == name]
            count = len(matching)
            duration = sum(float(event.get("dur", 0.0)) for event in matching)
            structural_counts[name].append(count)
            structural_us[name].append(duration)
            structural[name] = {"count": count, "cpu_total_us": duration}

        per_iteration.append(
            {
                "annotation": iteration.get("name"),
                "cpu_range_us": float(iteration.get("dur", 0.0)),
                "visible_device_us": sum(
                    float(event.get("dur", 0.0)) for event in owned_kernels
                ),
                "visible_kernel_count": len(owned_kernels),
                "visible_device_us_by_bucket": dict(iteration_buckets),
                "structural_driver_calls": structural,
            }
        )

    cpu_range_values = [row["cpu_range_us"] for row in per_iteration]
    device_values = [row["visible_device_us"] for row in per_iteration]
    return {
        "protocol": "qwen36-e190-bounded-xpu-trace-v1",
        "trace": str(path),
        "iteration_count": len(iterations),
        "iteration_cpu_range_us": stats(cpu_range_values),
        "visible_device_us_per_iteration": stats(device_values),
        "structural_driver_calls_per_iteration": {
            name: {
                "count": stats([float(value) for value in structural_counts[name]]),
                "cpu_total_us": stats(structural_us[name]),
            }
            for name in structural_names
        },
        "visible_device_buckets": [
            {
                "bucket": bucket,
                "kernel_count": bucket_count[bucket],
                "device_total_us": bucket_us[bucket],
                "device_mean_us_per_iteration": (
                    bucket_us[bucket] / len(iterations) if iterations else 0.0
                ),
            }
            for bucket in sorted(bucket_us, key=bucket_us.get, reverse=True)
        ],
        "top_visible_kernels": [
            {
                "kernel": name,
                "count": kernel_name_count[name],
                "device_total_us": duration,
                "device_mean_us_per_iteration": (
                    duration / len(iterations) if iterations else 0.0
                ),
            }
            for name, duration in kernel_name_us.most_common(30)
        ],
        "per_iteration": per_iteration,
        "interpretation_guard": (
            "Visible device time excludes kernels hidden inside XPUGraph replay; "
            "do not subtract it from synchronized model-forward time as if it "
            "were a complete device ledger."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    for report in reports:
        cpu = report["iteration_cpu_range_us"]
        device = report["visible_device_us_per_iteration"]
        print(f"TRACE -> {report['trace']}")
        print(
            "ITERATION -> "
            f"count={report['iteration_count']} "
            f"cpu_median_ms={cpu['median'] / 1000.0:.6f} "
            f"visible_device_mean_ms={device['mean'] / 1000.0:.6f}"
        )
        for name, row in report["structural_driver_calls_per_iteration"].items():
            print(
                "DRIVER -> "
                f"{name} count_median={row['count']['median']:.1f} "
                f"cpu_mean_us={row['cpu_total_us']['mean']:.3f}"
            )
        for row in report["visible_device_buckets"]:
            print(
                "DEVICE -> "
                f"{row['bucket']} count={row['kernel_count']} "
                f"mean_ms_per_iteration="
                f"{row['device_mean_us_per_iteration'] / 1000.0:.6f}"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
