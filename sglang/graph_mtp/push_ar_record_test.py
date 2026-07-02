#!/usr/bin/env python3
# push_ar_record_test.py -- decisive probe: does ar_allreduce_graph (K.6 capturable push-AR) RECORD
# into a raw torch.xpu.XPUGraph capture and REPLAY correctly, 2 ranks, in the sglang image?
# (Runs 5/20/22 hang at serve capture with push-AR engaged; isolate whether the .so records at all here.)
import ctypes
import os
import sys

import torch
import torch.distributed as dist

SO = os.environ.get("PUSH_SO", "/work/push_ar/libxpu_push_ar_graph.so")
MODE = int(os.environ.get("AR_MODE", "-1"))  # -1 = original entry; 0/1/2/3 = bisect modes


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29655"
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)
    lib = ctypes.CDLL(SO)
    lib.ar_setup_torch.restype = ctypes.c_int
    lib.ar_setup_torch.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_long]
    lib.ar_exchange.restype = ctypes.c_int
    lib.ar_exchange.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.ar_allreduce_ptr_dt.argtypes = [ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
    lib.ar_allreduce_graph.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int]
    if MODE >= 0:
        lib.ar_allreduce_graph_mode.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int, ctypes.c_int]
    if MODE == 4:
        lib.ar_graph_spin_init.restype = ctypes.c_int
        lib.ar_graph_spin_init.argtypes = [ctypes.c_long]
        lib.ar_allreduce_graph_spin.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_long, ctypes.c_int, ctypes.c_long]
    dev = f"xpu:{rank}"
    x = torch.full((5120,), float(rank + 1), device=dev, dtype=torch.bfloat16)

    q0 = torch.xpu.current_stream().sycl_queue
    rc = lib.ar_setup_torch(rank, ctypes.c_ulonglong(q0), 64 << 20)
    assert rc == 0, f"setup rc={rc}"
    rc = lib.ar_exchange(rank, b"/tmp/push_ar_rec_test.sock")
    assert rc == 0, f"exchange rc={rc}"
    print(f"[r{rank}] setup+exchange ok", flush=True)

    # eager path sanity
    y = x.clone()
    lib.ar_allreduce_ptr_dt(ctypes.c_ulonglong(y.data_ptr()), y.numel() * 2, 1)
    torch.xpu.synchronize()
    print(f"[r{rank}] eager push-AR ok sum={float(y[0])}", flush=True)

    if MODE == 4:
        rc = lib.ar_graph_spin_init(64 << 20)
        assert rc == 0
        print(f"[r{rank}] spin flags zeroed", flush=True)
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        z = x.clone()
        torch.xpu.synchronize()
        dist.barrier()
        g = torch.xpu.XPUGraph()
        g.capture_begin()
        z.add_(1.0)
        q = torch.xpu.current_stream().sycl_queue
        print(f"[r{rank}] recording ar_allreduce_graph (mode={MODE})...", flush=True)
        if MODE == 4:
            lib.ar_allreduce_graph_spin(ctypes.c_ulonglong(q), ctypes.c_ulonglong(z.data_ptr()), z.numel() * 2, 1, 64 << 20)
        elif MODE >= 0:
            lib.ar_allreduce_graph_mode(ctypes.c_ulonglong(q), ctypes.c_ulonglong(z.data_ptr()), z.numel() * 2, 1, MODE)
        else:
            lib.ar_allreduce_graph(ctypes.c_ulonglong(q), ctypes.c_ulonglong(z.data_ptr()), z.numel() * 2, 1)
        print(f"[r{rank}] recorded", flush=True)
        z.mul_(2.0)
        g.capture_end()
        print(f"[r{rank}] capture_end ok", flush=True)
        dist.barrier()
        for it in range(5):
            g.replay()
            torch.xpu.synchronize()
            print(f"[r{rank}] replay {it} ok z0={float(z[0])}", flush=True)
            dist.barrier()
    dist.destroy_process_group()
    print(f"[r{rank}] PASS", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(run, args=(2,), nprocs=2, join=True)
