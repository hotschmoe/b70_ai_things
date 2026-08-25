#!/usr/bin/env python3
"""Validate the Qwen3.6 TP2 oneCCL all-reduce graph contract.

Adapted from Steve Seguin's public-domain graph_allreduce_probe.py in
b70-optimization-lab at revision 523ca95b925308391707624530c29359edd05b6a.
This version runs direct and graph checks in one process-group lifetime and
records the loaded runtime identity for the local B70 reproduction campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist


def input_for_iteration(
    base: torch.Tensor, *, rank: int, iteration: int
) -> torch.Tensor:
    # Small integer BF16 values keep the reduction oracle exactly representable.
    return base + rank * 3 + (iteration % 17)


def expected_for_iteration(
    base: torch.Tensor, *, world_size: int, iteration: int
) -> torch.Tensor:
    rank_offset_sum = 3 * world_size * (world_size - 1) // 2
    return base * world_size + rank_offset_sum + world_size * (iteration % 17)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_library_paths(name: str) -> list[str]:
    return sorted(
        {
            line.rsplit(maxsplit=1)[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if name in line and line.rsplit(maxsplit=1)[-1].startswith("/")
        }
    )


def validate_mode(
    *,
    mode: str,
    iterations: int,
    base: torch.Tensor,
    static_input: torch.Tensor,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    graph: torch.xpu.XPUGraph | None = None
    if mode == "graph":
        graph = torch.xpu.XPUGraph()
        static_input.copy_(input_for_iteration(base, rank=rank, iteration=0))
        torch.xpu.synchronize()
        dist.barrier()
        with torch.xpu.graph(graph):
            work = dist.all_reduce(static_input, async_op=True)
            work.wait()
        torch.xpu.synchronize()
        dist.barrier()

    first_mismatch: dict[str, Any] | None = None
    mismatch_iterations = 0
    max_abs_diff = 0.0
    start = time.perf_counter()
    for iteration in range(iterations):
        static_input.copy_(
            input_for_iteration(base, rank=rank, iteration=iteration)
        )
        torch.xpu.synchronize()
        if graph is None:
            dist.all_reduce(static_input)
        else:
            graph.replay()
        torch.xpu.synchronize()

        expected = expected_for_iteration(
            base, world_size=world_size, iteration=iteration
        )
        if not torch.equal(static_input, expected):
            mismatch_iterations += 1
            diff = (static_input.float() - expected.float()).abs()
            iteration_max = float(diff.max().item())
            max_abs_diff = max(max_abs_diff, iteration_max)
            if first_mismatch is None:
                mismatch_flat = diff.reshape(-1).nonzero().reshape(-1)
                first_index = int(mismatch_flat[0].item())
                first_mismatch = {
                    "iteration": iteration,
                    "flat_index": first_index,
                    "actual": float(static_input.reshape(-1)[first_index].item()),
                    "expected": float(expected.reshape(-1)[first_index].item()),
                    "max_abs_diff": iteration_max,
                    "mismatch_elements": int(mismatch_flat.numel()),
                }

    torch.xpu.synchronize()
    return {
        "mode": mode,
        "iterations": iterations,
        "mismatch_iterations": mismatch_iterations,
        "first_mismatch": first_mismatch,
        "max_abs_diff": max_abs_diff,
        "avg_iteration_ms_including_sync_and_validation": (
            (time.perf_counter() - start) * 1000.0 / iterations
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--direct-iterations", type=int, default=256)
    parser.add_argument("--graph-iterations", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"this oracle requires world_size=2, got {world_size}")

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")
    device = torch.device(f"xpu:{local_rank}")
    base = (
        torch.arange(
            args.rows * args.hidden_size, dtype=torch.int32, device=device
        )
        .remainder_(31)
        .to(torch.bfloat16)
        .reshape(args.rows, args.hidden_size)
    )
    static_input = torch.empty_like(base)

    for iteration in range(args.warmup):
        static_input.copy_(
            input_for_iteration(base, rank=rank, iteration=iteration)
        )
        dist.all_reduce(static_input)
    torch.xpu.synchronize()
    dist.barrier()

    checks = []
    for mode, iterations in (
        ("direct", args.direct_iterations),
        ("graph", args.graph_iterations),
    ):
        checks.append(
            validate_mode(
                mode=mode,
                iterations=iterations,
                base=base,
                static_input=static_input,
                rank=rank,
                world_size=world_size,
            )
        )
        dist.barrier()

    ccl_paths = loaded_library_paths("libccl.so")
    ccl_files = [Path(path) for path in ccl_paths]
    kernels_path = Path(os.environ.get("CCL_KERNEL_PATH", "")) / "kernels.spv"
    expected_ccl_sha256 = os.environ.get("EXPECTED_CCL_SHA256")
    expected_kernels_sha256 = os.environ.get("EXPECTED_KERNELS_SHA256")
    loaded_ccl = [
        {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in ccl_files
    ]
    ccl_kernels = (
        {
            "path": str(kernels_path),
            "size": kernels_path.stat().st_size,
            "sha256": sha256_file(kernels_path),
        }
        if kernels_path.is_file()
        else None
    )
    identity_passed = bool(loaded_ccl) and ccl_kernels is not None
    if expected_ccl_sha256 is not None:
        identity_passed = identity_passed and all(
            item["sha256"] == expected_ccl_sha256 for item in loaded_ccl
        )
    if expected_kernels_sha256 is not None:
        identity_passed = (
            identity_passed
            and ccl_kernels is not None
            and ccl_kernels["sha256"] == expected_kernels_sha256
        )
    local_result = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "shape": [args.rows, args.hidden_size],
        "dtype": "bfloat16",
        "checks": checks,
        "loaded_ccl": loaded_ccl,
        "ccl_kernels": ccl_kernels,
        "expected_ccl_sha256": expected_ccl_sha256,
        "expected_kernels_sha256": expected_kernels_sha256,
        "identity_passed": identity_passed,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CCL_ATL_TRANSPORT",
                "CCL_TOPO_P2P_ACCESS",
                "CCL_ZE_IPC_EXCHANGE",
                "CCL_WORKER_COUNT",
                "FI_TCP_IFACE",
                "CCL_KVS_IFACE",
                "ONEAPI_DEVICE_SELECTOR",
                "ZE_AFFINITY_MASK",
            )
        },
    }
    all_results: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(all_results, local_result)
    passed = all(
        result is not None
        and result["identity_passed"]
        and all(check["mismatch_iterations"] == 0 for check in result["checks"])
        for result in all_results
    )

    if rank == 0:
        document = {
            "passed": passed,
            "backend": "xccl",
            "world_size": world_size,
            "source_attribution": (
                "Adapted from Steve Seguin's public-domain "
                "b70-optimization-lab graph_allreduce_probe.py"
            ),
            "ranks": all_results,
        }
        rendered = json.dumps(document, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")

    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
