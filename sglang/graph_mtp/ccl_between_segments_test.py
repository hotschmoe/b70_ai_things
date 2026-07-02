#!/usr/bin/env python3
# ccl_between_segments_test.py -- minimal 2-rank repro probe for the breakable-capture segfault
# (JOURNAL 2026-07-02 runs 7/9): does an EAGER oneCCL collective (dist.all_reduce / all_gather)
# issued on the capture stream BETWEEN torch.xpu.XPUGraph segment captures corrupt the next
# capture_begin/capture_end (segfault at final capture_end)?
#
# Pattern per rank (mirrors BreakableCudaGraphBackend + our eager_on_graph collective wrap):
#   seg1: capture(pool) [x*2]  ->  eager dist.all_reduce(x)  ->  seg2: capture(pool) [x+1]
#   -> eager all_gather -> seg3 capture EMPTY -> end -> replay all
# Run: torchrun-style spawn inside the sglang image with xccl (see ccl_between_segments_test.sh).
import os
import sys

import torch
import torch.distributed as dist


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29631"
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    dev = f"xpu:{rank}"
    x = torch.ones(5120, device=dev, dtype=torch.bfloat16)
    big = torch.ones(11 * 5120, device=dev, dtype=torch.bfloat16)

    pool = torch.xpu.graph_pool_handle()
    s = torch.xpu.Stream()
    print(f"[r{rank}] setup done", flush=True)

    with torch.xpu.stream(s):
        # warmup: one eager AR before any capture (like the runner warmup forwards)
        dist.all_reduce(x)
        torch.xpu.synchronize()
        print(f"[r{rank}] warmup AR ok", flush=True)

        g1 = torch.xpu.XPUGraph()
        g1.capture_begin(pool=pool)
        y = x * 2
        g1.capture_end()
        print(f"[r{rank}] seg1 ok", flush=True)

        # eager collective between segments, on the capture stream (the suspect)
        dist.all_reduce(y)
        print(f"[r{rank}] between-AR enqueued", flush=True)

        g2 = torch.xpu.XPUGraph()
        g2.capture_begin(pool=pool)
        z = y + 1
        g2.capture_end()
        print(f"[r{rank}] seg2 ok", flush=True)

        # eager all_gather between segments (the logits-gather analog)
        out = torch.empty(world * big.numel(), device=dev, dtype=big.dtype)
        dist.all_gather_into_tensor(out, big)
        print(f"[r{rank}] between-AG enqueued", flush=True)

        g3 = torch.xpu.XPUGraph()
        g3.capture_begin(pool=pool)
        g3.capture_end()  # trailing EMPTY segment (the __exit__ analog)
        print(f"[r{rank}] seg3 (empty) ok", flush=True)

        for i, g in enumerate((g1, g2, g3)):
            g.replay()
        torch.xpu.synchronize()
        print(f"[r{rank}] replay ok y0={float(y[0])} z0={float(z[0])}", flush=True)

    dist.destroy_process_group()
    print(f"[r{rank}] PASS", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(int(sys.argv[1]), 2)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run, args=(2,), nprocs=2, join=True)
