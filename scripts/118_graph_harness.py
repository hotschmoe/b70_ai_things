#!/usr/bin/env python3
# 118_graph_harness.py -- prove the capturable push all-reduce (libxpu_push_ar_graph.so) records into TORCH's
# own XPUGraph capture and replays correctly, in the 2-process TP-worker topology. This is the decisive
# pre-serve de-risk for the DECODE-capture goal (P2P_GPU K.6): K.4 proved native-cmd injection into OUR
# command_graph; this proves it into torch.xpu.graph (== sycl command_graph, ATen/xpu/XPUGraph.h).
import ctypes, os, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SO=os.environ.get("SO","/tmp/libxpu_push_ar_graph.so"); SOCK="/tmp/ar_graph.sock"
DT={torch.float32:0, torch.bfloat16:1, torch.float16:2}

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29555")
    os.environ["ZE_AFFINITY_MASK"]="0,1"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.xpu.set_device(rank)
    warm = torch.ones(4, device=f"xpu:{rank}"); torch.xpu.synchronize()
    qaddr = torch.xpu.current_stream().sycl_queue

    lib=ctypes.CDLL(SO)
    lib.ar_setup_torch.restype=ctypes.c_int; lib.ar_setup_torch.argtypes=[ctypes.c_int,ctypes.c_ulonglong,ctypes.c_long]
    lib.ar_exchange.restype=ctypes.c_int; lib.ar_exchange.argtypes=[ctypes.c_int,ctypes.c_char_p]
    lib.ar_allreduce_ptr_dt.argtypes=[ctypes.c_ulonglong,ctypes.c_long,ctypes.c_int]
    lib.ar_allreduce_graph.argtypes=[ctypes.c_ulonglong,ctypes.c_ulonglong,ctypes.c_long,ctypes.c_int]

    MAXB=64<<20
    if lib.ar_setup_torch(rank, ctypes.c_ulonglong(qaddr), MAXB)!=0: print(f"[r{rank}] setup FAIL",flush=True); return
    dist.barrier()
    if lib.ar_exchange(rank, SOCK.encode())!=0: print(f"[r{rank}] exchange FAIL",flush=True); return
    dist.barrier()
    if rank==0: print("[setup+exchange OK]",flush=True)

    dtype=torch.bfloat16; dt=DT[dtype]
    fillv = 1.0 if rank==0 else 3.0

    # ---- (1) EAGER sanity: the new .so's host-barrier path still works (and IPC-event setup didn't break it)
    n=10240//2  # bf16 elems for 10KB
    nbytes=n*2
    t=torch.full((n,), fillv, device=f"xpu:{rank}", dtype=dtype); torch.xpu.synchronize()
    lib.ar_allreduce_ptr_dt(ctypes.c_ulonglong(t.data_ptr()), nbytes, dt); torch.xpu.synchronize()
    eager_ok = abs(t[0].item()-4.0)<1e-2
    dist.barrier()
    if rank==0: print(f"[eager] verify {'OK(4.0)' if eager_ok else 'BAD '+str(t[0].item())}",flush=True)

    # ---- (2) GRAPH capture + replay of the capturable all-reduce ----
    # stable buffer (graph records its data_ptr); fill before capture.
    tg=torch.full((n,), fillv, device=f"xpu:{rank}", dtype=dtype); torch.xpu.synchronize()
    dist.barrier()
    cap_supported=True
    g=torch.xpu.XPUGraph()
    try:
        # capture records (does NOT execute) the push+native-sync+reduce on torch's stream.
        with torch.xpu.graph(g):
            capturing = torch.xpu.is_current_stream_capturing()
            capq = torch.xpu.current_stream().sycl_queue   # the CAPTURE stream (differs from setup stream)
            lib.ar_allreduce_graph(ctypes.c_ulonglong(capq), ctypes.c_ulonglong(tg.data_ptr()), nbytes, dt)
        if rank==0: print(f"[capture] capturing={capturing} setupq={qaddr} capq={capq} same={qaddr==capq}",flush=True)
    except Exception as e:
        cap_supported=False
        print(f"[r{rank}] CAPTURE EXCEPTION: {e}",flush=True)
    dist.barrier()
    if not cap_supported:
        lib.ar_teardown(); dist.destroy_process_group(); return

    # replay N times, fill before each, verify the captured graph performs the all-reduce.
    bad=0; N=50
    for it in range(N):
        tg.fill_(fillv); torch.xpu.synchronize()
        dist.barrier()
        g.replay()
        torch.xpu.synchronize()
        v=tg[0].item(); vlast=tg[-1].item()
        if not (abs(v-4.0)<1e-2 and abs(vlast-4.0)<1e-2): bad+=1
        dist.barrier()
    if rank==0:
        print(f"[graph] replay verify {N-bad}/{N} OK -> {'PASS' if bad==0 else 'FAIL bad='+str(bad)}",flush=True)

    # timing: replay latency (per-token decode allreduce in a captured graph)
    for _ in range(10): g.replay()
    torch.xpu.synchronize(); dist.barrier()
    t0=time.perf_counter()
    for _ in range(300): g.replay()
    torch.xpu.synchronize()
    dt_us=(time.perf_counter()-t0)/300*1e6
    if rank==0: print(f"[graph] replay latency {dt_us:.2f} us/allreduce ({nbytes}B bf16 decode)",flush=True)

    lib.ar_teardown()
    if rank==0: print("DONE_GRAPH_HARNESS",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
