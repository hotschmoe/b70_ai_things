"""Cross-card all-reduce microbenchmark for 2x Intel B70 (xccl over host-staged PCIe, no P2P).
Directly quantifies the TP comms bottleneck: algbw/busbw + latency vs message size.
No public Battlemage all-reduce numbers exist -> this is a novel datapoint.
Launched as 2 ranks via torch.multiprocessing.spawn, each pinned to one XPU.
"""
import os, time
import torch

# Intel oneCCL / IPEX bindings may be needed to register the xccl backend.
try:
    import intel_extension_for_pytorch as ipex  # noqa
except Exception as e:
    print("(no ipex:", e, ")")
try:
    import oneccl_bindings_for_pytorch  # noqa
except Exception as e:
    print("(no oneccl_bindings:", e, ")")

import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"

    backend = os.environ.get("ARBACKEND", "xccl")
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    except Exception as e:
        if rank == 0:
            print(f"init_process_group(backend={backend}) FAILED: {type(e).__name__}: {str(e)[:300]}")
        return

    if rank == 0:
        print(f"\n=== all-reduce bench: world={world} backend={backend} ===")
        print(f"{'bytes':>12} {'iters':>6} {'lat_ms':>10} {'algbw_GB/s':>12} {'busbw_GB/s':>12}")

    # 4 KB up to 256 MB
    exps = list(range(12, 29))  # 2^12 .. 2^28 bytes
    for e in exps:
        nbytes = 1 << e
        n = nbytes // 4
        x = torch.ones(n, dtype=torch.float32, device=dev)
        # warmup
        for _ in range(5):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        dist.barrier()
        iters = 50 if nbytes <= (1 << 20) else (20 if nbytes <= (1 << 24) else 10)
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t0) / iters
        size = n * 4
        algbw = size / dt / 1e9
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            print(f"{size:12d} {iters:6d} {dt*1e3:10.3f} {algbw:12.2f} {busbw:12.2f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("torch", torch.__version__, "xpu count", torch.xpu.device_count())
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("=== all-reduce bench DONE ===")
