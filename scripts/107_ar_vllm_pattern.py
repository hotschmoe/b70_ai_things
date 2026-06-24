#!/usr/bin/env python3
# 107_ar_vllm_pattern.py -- exercise the EXACT vLLM XpuCommunicator.all_reduce pattern with the custom op:
# bf16 tensors, real hidden=5120 prefill/decode shapes, clone-then-allreduce semantics. Proves the op is a
# correct + fast drop-in for `out=input.clone(); dist.all_reduce(out); return out`.
import ctypes, os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SO="/tmp/libxpu_push_ar_torch.so"; SOCK="/tmp/ar_vllm.sock"
H=5120
SHAPES=[(1,"decode b1"),(8,"decode b8"),(128,"prefill 128"),(2048,"prefill 2048"),(4096,"prefill 4096")]
MAXB=(4096*H*2)  # bf16 worst case

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29503")
    os.environ["ZE_AFFINITY_MASK"]="0,1"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.xpu.set_device(rank)
    _=torch.ones(4,device=f"xpu:{rank}"); torch.xpu.synchronize()
    qaddr=torch.xpu.current_stream().sycl_queue
    lib=ctypes.CDLL(SO)
    lib.ar_setup_torch.restype=ctypes.c_int; lib.ar_setup_torch.argtypes=[ctypes.c_int,ctypes.c_ulonglong,ctypes.c_long]
    lib.ar_exchange.restype=ctypes.c_int; lib.ar_exchange.argtypes=[ctypes.c_int,ctypes.c_char_p]
    lib.ar_allreduce_ptr_dt.argtypes=[ctypes.c_ulonglong,ctypes.c_long,ctypes.c_int]
    if lib.ar_setup_torch(rank,ctypes.c_ulonglong(qaddr),MAXB)!=0: print(f"[r{rank}] setup FAIL",flush=True); return
    dist.barrier()
    if lib.ar_exchange(rank,SOCK.encode())!=0: print(f"[r{rank}] exch FAIL",flush=True); return
    dist.barrier()

    DT=1  # bf16
    def custom_all_reduce(inp):  # mirrors XpuCommunicator.all_reduce
        out=inp.clone()
        lib.ar_allreduce_ptr_dt(ctypes.c_ulonglong(out.data_ptr()), out.numel()*out.element_size(), DT)
        return out

    if rank==0: print(f"{'shape':16s} {'lat us':>10s} {'GB/s':>8s} {'verify':>16s}",flush=True)
    fillv=1.0 if rank==0 else 3.0
    for tok,label in SHAPES:
        inp=torch.full((tok,H), fillv, device=f"xpu:{rank}", dtype=torch.bfloat16); torch.xpu.synchronize()
        out=custom_all_reduce(inp); torch.xpu.synchronize()
        # correctness: out==4.0 everywhere; input untouched (clone semantics)
        okv = torch.allclose(out.float(), torch.full_like(out.float(),4.0), atol=1e-2)
        okin = torch.allclose(inp.float(), torch.full_like(inp.float(),fillv), atol=1e-2)
        nbytes=inp.numel()*2
        iters=300 if tok<=8 else (100 if tok<=128 else 40)
        for _ in range(5): custom_all_reduce(inp)
        dist.barrier(); t0=time.perf_counter()
        for _ in range(iters): custom_all_reduce(inp)
        torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/iters
        if rank==0:
            v="OK" if (okv and okin) else f"BAD(out={out.float().mean():.2f},in={inp.float().mean():.2f})"
            print(f"{label:16s} {dt*1e6:10.2f} {nbytes/dt/1e9:8.2f} {v:>16s}",flush=True)
    lib.ar_teardown()
    if rank==0: print("DONE_AR_VLLM",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    mp.set_start_method("spawn",force=True)
    mp.spawn(worker,args=(2,),nprocs=2,join=True)
