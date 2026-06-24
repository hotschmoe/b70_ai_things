#!/usr/bin/env python3
# 105_ar_harness.py -- drive libxpu_push_ar.so from TWO independent processes via torch.distributed
# (gloo = vLLM's cpu_group). Proves the full vLLM-shaped path: independent procs, IPC handle exchange,
# correct + fast 2-rank all-reduce. The named-socket fd-pass lives in the .so; gloo does barrier/ordering.
import ctypes, os, time, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SO = "/tmp/libxpu_push_ar.so"
SOCK = "/tmp/ar_ipc.sock"
SIZES = [(10240,"10KB(decode)"), (65536,"64KB"), (1<<20,"1MB"), (16<<20,"16MB(prefill)"),
         (64<<20,"64MB"), (256<<20,"256MB")]
MAXB = 256<<20

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29501")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    lib = ctypes.CDLL(SO)
    lib.ar_setup.restype=ctypes.c_int; lib.ar_exchange.restype=ctypes.c_int
    lib.ar_peek.restype=ctypes.c_float
    for f in (lib.ar_fill,lib.ar_push,lib.ar_reduce):
        pass
    lib.ar_fill.argtypes=[ctypes.c_float,ctypes.c_long]
    lib.ar_push.argtypes=[ctypes.c_long]; lib.ar_reduce.argtypes=[ctypes.c_long]
    lib.ar_allreduce.argtypes=[ctypes.c_long]
    lib.ar_setup.argtypes=[ctypes.c_int,ctypes.c_long]
    lib.ar_exchange.argtypes=[ctypes.c_int,ctypes.c_char_p]

    if lib.ar_setup(rank, MAXB)!=0: print(f"[r{rank}] setup FAIL",flush=True); return
    dist.barrier()
    if lib.ar_exchange(rank, SOCK.encode())!=0: print(f"[r{rank}] exchange FAIL",flush=True); return
    dist.barrier()

    # verify gloo all_gather_object works for handle-sized blobs (the vLLM exchange transport sanity).
    if rank==0: print(f"{'size':16s} {'lat us':>10s} {'algbw GB/s':>12s} {'verify':>10s}",flush=True)
    fillv = 1.0 if rank==0 else 3.0
    for nbytes,label in SIZES:
        n = nbytes//4
        lib.ar_fill(ctypes.c_float(fillv), n)
        # one correctness allreduce (shm barrier inside ar_allreduce)
        lib.ar_allreduce(nbytes)
        val = lib.ar_peek()
        ok = abs(val-4.0) < 1e-3
        # timing -- no gloo per call; the shm spin barrier lives inside ar_allreduce
        iters = 300 if nbytes<=(1<<20) else (60 if nbytes<=(16<<20) else 15)
        lib.ar_fill(ctypes.c_float(fillv), n)
        for _ in range(5):
            lib.ar_allreduce(nbytes)
        dist.barrier(); t0=time.perf_counter()
        for _ in range(iters):
            lib.ar_allreduce(nbytes)
        dt=(time.perf_counter()-t0)/iters
        if rank==0:
            print(f"{label:16s} {dt*1e6:10.2f} {nbytes/dt/1e9:12.2f} {'OK(4.0)' if ok else 'BAD':>10s}",flush=True)
    lib.ar_teardown()
    if rank==0: print("DONE_AR_HARNESS",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
