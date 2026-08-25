#!/usr/bin/env python3
"""Run one June profile-size fused MoE immediately before TP=2 oneCCL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch
import torch.distributed as dist

from qwen36_june_fused_moe_single import fused_call, make_inputs, tensor_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"expected world size 2, got {world_size}")

    torch.xpu.set_device(local_rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    import vllm_xpu_kernels._xpu_C as xpu_module

    local: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "rows": args.rows,
        "stages": [],
    }
    expected_xpu = os.environ["EXPECTED_XPU_C_SHA256"]
    xpu_path = Path(xpu_module.__file__).resolve()
    local["runtime"] = {
        "torch": torch.__version__,
        "xpu_c": str(xpu_path),
        "xpu_c_sha256": sha256_file(xpu_path),
        "expected_xpu_c_sha256": expected_xpu,
    }
    if local["runtime"]["xpu_c_sha256"] != expected_xpu:
        raise RuntimeError(f"June _xpu_C identity mismatch: {local['runtime']}")

    exit_code = 1
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(backend="xccl")
            initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                pipeline_model_parallel_size=1,
                backend="xccl",
            )
            warmup = torch.ones(1, dtype=torch.float32, device=f"xpu:{local_rank}")
            dist.all_reduce(warmup)
            torch.xpu.synchronize()
            local["stages"].append("process_group_warmup")
            print(f"rank={rank} stage=process_group_warmup OK", flush=True)

            group = get_tp_group()
            if not group.use_custom_op_call:
                raise RuntimeError("vLLM custom-op collective route is inactive")
            compiled_reduce = torch.compile(
                tensor_model_parallel_all_reduce,
                fullgraph=True,
                dynamic=True,
            )
            small = torch.full(
                (1, 2048), rank + 1, dtype=torch.bfloat16, device=f"xpu:{local_rank}"
            )
            for _ in range(2):
                reduced = compiled_reduce(small)
            torch.xpu.synchronize()
            dist.barrier()
            if not torch.all(reduced == 3):
                raise RuntimeError("compiled collective warmup mismatch")
            local["stages"].append("compiled_collective_warmup")
            print(f"rank={rank} stage=compiled_collective_warmup OK", flush=True)

            inputs = make_inputs(args.rows, 20260825 + rank * 1000)
            moe_output = torch.empty_like(inputs["hidden_states"])
            for iteration in range(2):
                fused_call(inputs, moe_output)
                torch.xpu.synchronize()
                print(
                    f"rank={rank} stage=fused_moe_profile iteration={iteration} OK",
                    flush=True,
                )
            local["moe_sha256"] = tensor_sha256(moe_output)
            local["stages"].append("fused_moe_profile")
            dist.barrier()

            reduced = compiled_reduce(moe_output)
            torch.xpu.synchronize()
            local["collective_sha256"] = tensor_sha256(reduced)
            local["collective_finite"] = bool(torch.isfinite(reduced).all().cpu())
            local["stages"].append("compiled_collective_after_moe")
            print(f"rank={rank} stage=compiled_collective_after_moe OK", flush=True)

            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, local)
            passed = all(
                item is not None
                and item.get("collective_finite")
                and item.get("collective_sha256") == gathered[0].get("collective_sha256")
                for item in gathered
            )
            if rank == 0:
                document = {
                    "protocol": "qwen36-june-tp2-moe-collective-boundary-v1",
                    "passed": passed,
                    "ranks": gathered,
                }
                args.output.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="ascii",
                )
                print(json.dumps(document, sort_keys=True), flush=True)
            exit_code = 0 if passed else 1
    except BaseException as exc:
        local["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        partial = args.output.with_name(f"{args.output.stem}.rank{rank}.partial.json")
        partial.write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
        print(f"rank={rank} failure={local['exception']}", file=sys.stderr, flush=True)
        raise
    finally:
        for cleanup in (destroy_model_parallel, destroy_distributed_environment):
            try:
                cleanup()
            except BaseException as exc:
                print(
                    f"rank={rank} cleanup={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
