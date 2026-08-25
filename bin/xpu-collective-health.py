#!/usr/bin/env python3
"""Two-rank compiled XCCL health probe for the B70 reset guard."""

from __future__ import annotations

from datetime import timedelta
import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol


def make_input(base: torch.Tensor, rank: int, iteration: int) -> torch.Tensor:
    return base + rank * 3 + (iteration % 7)


def expected(base: torch.Tensor, world_size: int, iteration: int) -> torch.Tensor:
    rank_sum = 3 * world_size * (world_size - 1) // 2
    return base * world_size + rank_sum + world_size * (iteration % 7)


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"expected world_size=2, got {world_size}")

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl", timeout=timedelta(seconds=45))
    try:
        device = torch.device(f"xpu:{local_rank}")
        base = (
            torch.arange(4 * 5120, dtype=torch.int32, device=device)
            .remainder_(31)
            .to(torch.bfloat16)
            .reshape(4, 5120)
        )

        eager = make_input(base, rank, 0)
        dist.all_reduce(eager)
        torch.xpu.synchronize()
        if not torch.equal(eager, expected(base, world_size, 0)):
            raise RuntimeError("eager XCCL all-reduce mismatch")

        def compiled_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
            reduced = funcol.all_reduce(tensor, "sum", dist.group.WORLD)
            return funcol.wait_tensor(reduced)

        compiled = torch.compile(
            compiled_all_reduce, fullgraph=True, dynamic=False
        )
        for iteration in range(2):
            output = compiled(make_input(base, rank, iteration))
            torch.xpu.synchronize()
            if not torch.equal(output, expected(base, world_size, iteration)):
                raise RuntimeError(
                    f"compiled XCCL warmup mismatch at iteration {iteration}"
                )
        for iteration in range(2, 10):
            output = compiled(make_input(base, rank, iteration))
            torch.xpu.synchronize()
            if not torch.equal(output, expected(base, world_size, iteration)):
                raise RuntimeError(
                    f"compiled XCCL mismatch at iteration {iteration}"
                )

        passed = [None] * world_size
        dist.all_gather_object(passed, True)
        if rank == 0 and all(value is True for value in passed):
            print(
                "COLLECTIVE_HEALTH_OK "
                f"world_size={world_size} shape=4x5120 compiled_iterations=10 "
                f"p2p={os.environ.get('CCL_TOPO_P2P_ACCESS')}"
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        return_code = main()
    except Exception as error:
        print(
            "COLLECTIVE_HEALTH_FAIL "
            f"rank={os.environ.get('RANK')} type={type(error).__name__} "
            f"message={str(error)[:500]}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)
    sys.exit(return_code)
