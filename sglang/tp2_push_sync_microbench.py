#!/usr/bin/env python3
"""Two-B70 ordered-sequence microbench for TP=2 collective sync paths."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import statistics
import time

import torch
try:
    import intel_extension_for_pytorch  # noqa: F401
except Exception:
    pass
try:
    import oneccl_bindings_for_pytorch  # noqa: F401
except Exception:
    pass
import torch.distributed as dist
import torch.multiprocessing as mp


DT_BF16 = 1


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[pos]


def load_current(path: str, rank: int, queue: int, max_bytes: int, sock_path: str):
    lib = ctypes.CDLL(path)
    lib.ar_setup_torch.restype = ctypes.c_int
    lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
    lib.ar_exchange.restype = ctypes.c_int
    lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_allreduce_ptr_dt.argtypes = [ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
    rc = lib.ar_setup_torch(rank, ctypes.c_ulonglong(queue), max_bytes)
    if rc == 0:
        rc = lib.ar_exchange(rank, sock_path.encode())
    if rc:
        raise RuntimeError(f"current setup rc={rc}")
    return lib


def load_async(path: str, rank: int, queue: int, max_bytes: int, sock_path: str, ring: int):
    lib = ctypes.CDLL(path)
    lib.ar_ea_setup.restype = ctypes.c_int
    lib.ar_ea_setup.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_long]
    lib.ar_ea_exchange.restype = ctypes.c_int
    lib.ar_ea_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_allreduce_eager_async.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.c_long,
        ctypes.c_int,
    ]
    lib.ar_ea_teardown.argtypes = []
    rc = lib.ar_ea_setup(rank, ctypes.c_ulonglong(queue), ring, max_bytes)
    if rc == 0:
        rc = lib.ar_ea_exchange(rank, sock_path.encode())
    if rc:
        raise RuntimeError(f"async setup rc={rc}")
    return lib


def worker(rank: int, world: int, args, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world)
    queue = int(torch.xpu.current_stream().sycl_queue)
    max_bytes = max(args.rows) * args.hidden * 2
    lib = None
    if args.mode == "current":
        lib = load_current(args.current_so, rank, queue, max_bytes, args.sock)
    elif args.mode == "async":
        lib = load_async(args.async_so, rank, queue, max_bytes, args.sock, args.ring)
    dist.barrier()

    rank_rows = []
    for rows in args.rows:
        n = rows * args.hidden
        x = torch.empty((rows, args.hidden), device=f"xpu:{rank}", dtype=torch.bfloat16)
        saved = torch.empty((args.calls, rows, args.hidden), device=x.device, dtype=x.dtype)
        times = []
        for rep in range(args.warmup + args.repeats):
            dist.barrier()
            torch.xpu.synchronize()
            start = time.perf_counter()
            for call in range(args.calls):
                sentinel = 1 + (call & 1)
                x.fill_(float(sentinel + rank))
                if args.inject_delay and rank == 1 and call and call % args.delay_every == 0:
                    time.sleep(args.inject_delay)
                if args.mode == "oneccl":
                    dist.all_reduce(x)
                elif args.mode == "current":
                    lib.ar_allreduce_ptr_dt(
                        ctypes.c_ulonglong(x.data_ptr()), x.numel() * x.element_size(), DT_BF16
                    )
                else:
                    lib.ar_allreduce_eager_async(
                        ctypes.c_ulonglong(queue), ctypes.c_ulonglong(x.data_ptr()),
                        x.numel() * x.element_size(), DT_BF16,
                    )
                saved[call].copy_(x)
            caller_ms = 1000.0 * (time.perf_counter() - start)
            torch.xpu.synchronize()
            total_ms = 1000.0 * (time.perf_counter() - start)
            expected = torch.tensor(
                [3.0 if not (i & 1) else 5.0 for i in range(args.calls)],
                device=x.device,
                dtype=x.dtype,
            ).view(args.calls, 1, 1)
            correct = bool(torch.all(saved == expected).item())
            if not correct:
                raise RuntimeError(f"rank {rank} rows {rows} rep {rep}: numerical mismatch")
            if rep >= args.warmup:
                times.append((caller_ms, total_ms))
        rank_rows.append({
            "rows": rows,
            "bytes": max_bytes if rows == max(args.rows) else n * 2,
            "caller_ms_median": statistics.median(x[0] for x in times),
            "total_ms_median": statistics.median(x[1] for x in times),
            "caller_us_per_call_median": statistics.median(x[0] for x in times) * 1000.0 / args.calls,
            "total_us_per_call_median": statistics.median(x[1] for x in times) * 1000.0 / args.calls,
            "total_us_per_call_p95_repeat": percentile([x[1] * 1000.0 / args.calls for x in times], 0.95),
            "correct": True,
        })

    dist.barrier()
    if args.mode == "async" and lib is not None:
        torch.xpu.synchronize()
        lib.ar_ea_teardown()
    result_queue.put((rank, rank_rows))
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oneccl", "current", "async"), required=True)
    ap.add_argument("--rows", default="1,11,44")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--calls", type=int, default=159)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--inject-delay", type=float, default=0.0002)
    ap.add_argument("--delay-every", type=int, default=37)
    ap.add_argument("--current-so", default="/push/libxpu_push_ar_graph.so")
    ap.add_argument("--async-so", default="/push/libxpu_push_ar_eager.so")
    ap.add_argument("--sock", default=f"/tmp/b70_sync_{os.getpid()}.sock")
    ap.add_argument("--master-port", type=int, default=29617)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.rows = [int(x) for x in args.rows.split(",")]
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    mp.spawn(worker, args=(2, args, result_queue), nprocs=2, join=True)
    results = dict(result_queue.get() for _ in range(2))
    summary = {
        "mode": args.mode,
        "async_hostwait_input": os.environ.get("ASYNC_HOSTWAIT_INPUT", "0"),
        "calls_per_sequence": args.calls,
        "hidden": args.hidden,
        "rows": args.rows,
        "rank_results": results,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
