#!/usr/bin/env python3
"""Exercise push-AR through June vLLM's real TP communicator lifecycle."""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    torch.xpu.set_device(rank)
    torch.ones(1, device=f"xpu:{rank}")

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        init_model_parallel_group,
    )
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
            distributed_executor_backend="mp",
        )
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:29671",
            local_rank=rank,
            backend="xccl",
        )
        tp = init_model_parallel_group(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            backend="xccl",
            group_name="tp",
        )
    comm = tp.device_communicator
    if comm is None:
        raise RuntimeError("TP group has no device communicator")
    if not getattr(comm, "_push_ar_ready", False):
        raise RuntimeError(f"rank {rank} push-AR was not ready at TP init")

    fill = 1.0 if rank == 0 else 3.0
    graph_in = torch.full((5120,), fill, dtype=torch.bfloat16, device=f"xpu:{rank}")
    graph = torch.xpu.XPUGraph()
    print(f"[oracle r{rank}] entering graph capture", flush=True)
    with torch.xpu.graph(graph):
        print(
            f"[oracle r{rank}] capturing="
            f"{torch.xpu.is_current_stream_capturing()}",
            flush=True,
        )
        graph_out = comm.all_reduce(graph_in)
    print(f"[oracle r{rank}] graph capture complete", flush=True)

    bad = 0
    replay_count = 64
    start = time.perf_counter()
    for _ in range(replay_count):
        graph_in.fill_(fill)
        torch.xpu.synchronize()
        graph.replay()
        torch.xpu.synchronize()
        if abs(graph_out[0].item() - 4.0) > 0.01:
            bad += 1
    elapsed = time.perf_counter() - start
    if bad:
        raise RuntimeError(f"rank {rank} graph mismatches: {bad}/{replay_count}")
    if rank == 0:
        print(
            "PUSH_AR_INIT_ORACLE_PASS "
            f"graph={replay_count}/{replay_count} "
            f"wall_us_per_replay={elapsed / replay_count * 1e6:.2f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
