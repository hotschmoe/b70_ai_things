#!/usr/bin/env python3
"""Two-rank XPU breakable-graph oracle with eager in-place oneCCL."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import sitecustomize
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper


def model(input_: torch.Tensor, rank: int) -> torch.Tensor:
    local = input_ * 2 + rank
    reduced = get_tp_group().all_reduce(local)
    return reduced * 3


def run_oracle() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(local_rank)
    init_distributed_environment(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="xccl",
    )
    ensure_model_parallel_initialized(2, 1)

    static_input = torch.empty((1, 5120), dtype=torch.bfloat16, device="xpu")
    static_input.fill_(rank)
    torch.xpu.synchronize()

    with _torch_cuda_wrapper():
        dist.barrier(group=get_tp_group().cpu_group)
        capture = BreakableCUDAGraphCapture(pool=torch.xpu.graph_pool_handle())
        with capture:
            output = model(static_input, rank)
        torch.xpu.synchronize()

    if capture.num_eager_breaks != 1 or capture.num_graphs != 2:
        raise RuntimeError(
            f"rank={rank} unexpected segments: graphs={capture.num_graphs} "
            f"eager={capture.num_eager_breaks}"
        )

    output_pointer = output.data_ptr()
    if output_pointer == static_input.data_ptr():
        raise RuntimeError(f"rank={rank} output aliases input")
    replay_bases = (0, 1, 7, 19) * 4
    for iteration, base in enumerate(replay_bases):
        dist.barrier(group=get_tp_group().cpu_group)
        static_input.fill_(base + rank)
        input_before = static_input.clone()
        with _torch_cuda_wrapper():
            capture.replay()
        torch.xpu.synchronize()
        expected = float(12 * base + 9)
        actual = output.float()
        if not torch.all(actual == expected):
            delta = float((actual - expected).abs().max().item())
            raise RuntimeError(
                f"rank={rank} iteration={iteration} expected={expected} "
                f"max_delta={delta}"
            )
        if not torch.equal(static_input, input_before):
            raise RuntimeError(f"rank={rank} iteration={iteration} input mutated")
        if output.data_ptr() != output_pointer:
            raise RuntimeError(f"rank={rank} iteration={iteration} output moved")
        if rank == 0:
            print(
                f"REPLAY_OK iteration={iteration} expected={expected:.1f} "
                f"mean={float(actual.mean().item()):.1f}",
                flush=True,
            )

    expected_calls = len(replay_bases) + 1
    if sitecustomize._b70_eager_ar_calls != expected_calls:
        raise RuntimeError(
            f"rank={rank} helper_calls={sitecustomize._b70_eager_ar_calls} "
            f"expected={expected_calls}"
        )
    dist.barrier(group=get_tp_group().cpu_group)
    if rank == 0:
        print(
            f"BREAKABLE_ALLREDUCE_OK world_size={dist.get_world_size()} "
            f"graphs={capture.num_graphs} eager={capture.num_eager_breaks} "
            f"helper_calls={sitecustomize._b70_eager_ar_calls}",
            flush=True,
        )
    destroy_model_parallel()
    destroy_distributed_environment()


def main() -> None:
    with set_current_vllm_config(VllmConfig()):
        run_oracle()


if __name__ == "__main__":
    main()
