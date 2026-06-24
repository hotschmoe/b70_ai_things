#!/usr/bin/env python3
# 106_ar_torch_harness.py -- all-reduce REAL torch.xpu tensors via libxpu_push_ar_torch.so, 2 procs.
# Proves the op runs in torch's L0 context (operating on a torch tensor's data_ptr). If verify passes,
# the only thing between here and a live vLLM serve is monkeypatching XpuCommunicator.all_reduce.
import ctypes, os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SO="/tmp/libxpu_push_ar_torch.so"; SOCK="/tmp/ar_torch.sock"
SIZES=[(10240,"10KB(decode)"),(65536,"64KB"),(1<<20,"1MB"),(16<<20,"16MB(prefill)"),(64<<20,"64MB")]
MAXB=64<<20

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29502")
    os.environ["ZE_AFFINITY_MASK"]="0,1"  # both visible; torch picks device by index
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.xpu.set_device(rank)
    # touch xpu so the runtime + stream exist, then grab torch's sycl::queue address.
    warm = torch.ones(4, device=f"xpu:{rank}"); torch.xpu.synchronize()
    qaddr = torch.xpu.current_stream().sycl_queue

    lib=ctypes.CDLL(SO)
    lib.ar_setup_torch.restype=ctypes.c_int; lib.ar_setup_torch.argtypes=[ctypes.c_int,ctypes.c_ulonglong,ctypes.c_long]
    lib.ar_exchange.restype=ctypes.c_int; lib.ar_exchange.argtypes=[ctypes.c_int,ctypes.c_char_p]
    lib.ar_allreduce_ptr.argtypes=[ctypes.c_ulonglong,ctypes.c_long]

    if lib.ar_setup_torch(rank, ctypes.c_ulonglong(qaddr), MAXB)!=0: print(f"[r{rank}] setup FAIL",flush=True); return
    dist.barrier()
    if lib.ar_exchange(rank, SOCK.encode())!=0: print(f"[r{rank}] exchange FAIL",flush=True); return
    dist.barrier()

    if rank==0: print(f"{'size':16s} {'lat us':>10s} {'algbw GB/s':>12s} {'verify':>14s}",flush=True)
    fillv = 1.0 if rank==0 else 3.0
    for nbytes,label in SIZES:
        n=nbytes//4
        t=torch.full((n,), fillv, device=f"xpu:{rank}", dtype=torch.float32); torch.xpu.synchronize()
        lib.ar_allreduce_ptr(ctypes.c_ulonglong(t.data_ptr()), nbytes)
        torch.xpu.synchronize()
        v=t[0].item(); vlast=t[-1].item()
        ok = abs(v-4.0)<1e-3 and abs(vlast-4.0)<1e-3   # torch tensor itself now holds the sum
        # timing
        iters=300 if nbytes<=(1<<20) else (60 if nbytes<=(16<<20) else 15)
        t.fill_(fillv); torch.xpu.synchronize()
        for _ in range(5): lib.ar_allreduce_ptr(ctypes.c_ulonglong(t.data_ptr()), nbytes)
        dist.barrier(); t0=time.perf_counter()
        for _ in range(iters): lib.ar_allreduce_ptr(ctypes.c_ulonglong(t.data_ptr()), nbytes)
        dt=(time.perf_counter()-t0)/iters
        if rank==0:
            print(f"{label:16s} {dt*1e6:10.2f} {nbytes/dt/1e9:12.2f} {('OK(4.0)' if ok else f'BAD({v},{vlast})'):>14s}",flush=True)
    lib.ar_teardown()
    if rank==0: print("DONE_AR_TORCH",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
