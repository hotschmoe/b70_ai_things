#!/usr/bin/env python3
"""Bounded two-rank oracle for vLLM's direct-P2P XPU communicator path."""

import os
import time

import torch
import torch.distributed as dist


def event(rank: int, name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"F06B_EVENT rank={rank} monotonic_ns={time.monotonic_ns()} name={name}{suffix}", flush=True)


def check_tensor(rank: int, label: str, value: torch.Tensor) -> None:
    expected = 7.0
    error = float((value.float() - expected).abs().max().cpu())
    if error != 0.0:
        raise RuntimeError(f"{label}: rank={rank} max_error={error}")
    event(rank, "verified", f"case={label} max_error={error}")


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"F06b requires world_size=2, got {world_size}")

    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl", init_method="env://")

    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        raise RuntimeError(f"vLLM selected unexpected platform: {current_platform}")

    event(rank, "coordinator-entry", f"local_rank={local_rank}")
    coordinator = GroupCoordinator(
        group_ranks=[[0, 1]],
        local_rank=local_rank,
        torch_distributed_backend="xccl",
        use_device_communicator=True,
        use_message_queue_broadcaster=False,
        group_name="f06b_tp",
    )
    communicator_name = type(coordinator.device_communicator).__name__
    if communicator_name != "XpuCommunicator":
        raise RuntimeError(f"unexpected communicator: {communicator_name}")
    event(rank, "coordinator-return", f"communicator={communicator_name}")

    def sync() -> None:
        dist.barrier(group=coordinator.cpu_group)

    direct_input = torch.full(
        (4, 5120), float(rank + 1), dtype=torch.bfloat16, device=f"xpu:{local_rank}"
    )
    sync()
    event(rank, "device-communicator-entry", "shape=4x5120")
    direct_output = coordinator.device_communicator.all_reduce(direct_input)
    direct_output = direct_output * 2.0 + 1.0
    torch.xpu.synchronize()
    event(rank, "device-communicator-return", "shape=4x5120")
    check_tensor(rank, "device-eager-4x5120", direct_output)

    def step(input_: torch.Tensor) -> torch.Tensor:
        reduced = coordinator.all_reduce(input_)
        return reduced * 2.0 + 1.0

    eager_input = torch.full(
        (4, 5120), float(rank + 1), dtype=torch.bfloat16, device=f"xpu:{local_rank}"
    )
    sync()
    event(rank, "custom-op-eager-entry", "shape=4x5120")
    eager_output = step(eager_input)
    torch.xpu.synchronize()
    event(rank, "custom-op-eager-return", "shape=4x5120")
    check_tensor(rank, "custom-op-eager-4x5120", eager_output)

    compiled_step = torch.compile(step, backend="inductor", fullgraph=True, dynamic=False)
    shapes = ((1, 5120), (4, 5120), (256, 5120), (2048, 5120))
    iterations = 10
    for rows, columns in shapes:
        input_ = torch.full(
            (rows, columns),
            float(rank + 1),
            dtype=torch.bfloat16,
            device=f"xpu:{local_rank}",
        )
        sync()
        event(rank, "compiled-case-entry", f"shape={rows}x{columns} iterations={iterations}")
        output = input_
        for iteration in range(iterations):
            event(rank, "compiled-call-entry", f"shape={rows}x{columns} iteration={iteration}")
            output = compiled_step(input_)
            torch.xpu.synchronize()
            event(rank, "compiled-call-return", f"shape={rows}x{columns} iteration={iteration}")
        check_tensor(rank, f"compiled-{rows}x{columns}", output)
        event(rank, "compiled-case-return", f"shape={rows}x{columns} iterations={iterations}")

    sync()
    event(rank, "oracle-pass", "eager=2 compiled_shapes=4 compiled_iterations=40")
    coordinator.destroy()
    dist.destroy_process_group()
    if rank == 0:
        print(
            "F06B_VLLM_COMMUNICATOR_OK world_size=2 communicator=XpuCommunicator "
            "eager=2 compiled_shapes=1x5120,4x5120,256x5120,2048x5120 "
            "compiled_iterations=40 p2p=1",
            flush=True,
        )


if __name__ == "__main__":
    main()
