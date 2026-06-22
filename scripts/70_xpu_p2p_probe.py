#!/usr/bin/env python3
# Direct B70<->B70 P2P probe -- measures the ACTUAL peer path, independent of vLLM/oneCCL.
# Peer-direct Gen3 x16 ~= 13-15 GB/s; host-staged round-trip ~= half that. So the bandwidth
# number tells us whether xpu:0<->xpu:1 d2d copy goes peer-direct or bounces through host RAM.
# Also reports a small-message ping-pong latency (the allreduce-relevant number; Seguin's raw
# allreduce was 15-17us). Run inside the :int8 image with BOTH cards exposed (ZE_AFFINITY_MASK=0,1).
import os, time, sys
import torch

def bw_test(src, dst, mb, iters=30):
    numel = mb * 1024 * 1024 // 2  # fp16 = 2 bytes
    a = torch.ones(numel, dtype=torch.float16, device=f"xpu:{src}")
    b = torch.empty(numel, dtype=torch.float16, device=f"xpu:{dst}")
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    for _ in range(5):  # warmup
        b.copy_(a)
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    t = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    dt = (time.perf_counter() - t) / iters
    gb = mb / 1024.0
    return dt, gb / dt

def pingpong_latency(src, dst, iters=200):
    # tiny copies back and forth -> per-copy latency (proxy for collective small-msg latency)
    a = torch.ones(8, dtype=torch.float16, device=f"xpu:{src}")
    b = torch.empty(8, dtype=torch.float16, device=f"xpu:{dst}")
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    for _ in range(20):
        b.copy_(a)
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    t = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.xpu.synchronize(src); torch.xpu.synchronize(dst)
    return (time.perf_counter() - t) / iters * 1e6  # us

def main():
    n = torch.xpu.device_count()
    print(f"xpu device_count = {n}")
    for i in range(n):
        try:
            print(f"  xpu:{i} = {torch.xpu.get_device_name(i)}")
        except Exception as e:
            print(f"  xpu:{i} name err: {e}")
    if n < 2:
        print("NEED >=2 xpu devices; abort"); return 1
    # env that may influence the peer path
    for k in ("ZE_AFFINITY_MASK","CCL_TOPO_P2P_ACCESS","CCL_ZE_IPC_EXCHANGE","SYCL_UR_USE_LEVEL_ZERO_V2"):
        print(f"  env {k} = {os.environ.get(k)}")
    print("=== d2d copy bandwidth xpu0 -> xpu1 (peer-direct ~13-15 GB/s Gen3; host-staged ~half) ===")
    for mb in (16, 64, 256):
        try:
            dt, gbs = bw_test(0, 1, mb)
            print(f"  {mb:>4} MB  xpu0->xpu1  {dt*1e3:8.3f} ms  {gbs:7.2f} GB/s")
        except Exception as e:
            print(f"  {mb} MB FAILED: {type(e).__name__}: {e}")
    print("=== reverse xpu1 -> xpu0 ===")
    for mb in (64,):
        try:
            dt, gbs = bw_test(1, 0, mb)
            print(f"  {mb:>4} MB  xpu1->xpu0  {dt*1e3:8.3f} ms  {gbs:7.2f} GB/s")
        except Exception as e:
            print(f"  {mb} MB FAILED: {type(e).__name__}: {e}")
    print("=== small-message ping-pong latency (8 elems) ===")
    try:
        lat = pingpong_latency(0, 1)
        print(f"  xpu0->xpu1 per-copy: {lat:.2f} us")
    except Exception as e:
        print(f"  ping-pong FAILED: {type(e).__name__}: {e}")
    print("DONE_P2P_PROBE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
