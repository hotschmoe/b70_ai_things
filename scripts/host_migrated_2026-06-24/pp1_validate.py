import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa
MODE=os.environ.get("MODE","?"); ITERS=int(os.environ.get("ITERS","200")); WARMUP=int(os.environ.get("WARMUP","40"))
BW=608.0; TOPS=367.0
print(f"MODE={MODE} torch={torch.__version__}", flush=True)
def mk(m,k,n,seed):
    g=torch.Generator().manual_seed(seed)
    xq=torch.randint(-127,128,(m,k),generator=g,dtype=torch.int8)
    xs=(torch.rand((m,1),generator=g,dtype=torch.float32)*0.02+0.005).to(torch.float16)
    wq=torch.randint(-127,128,(k,n),generator=g,dtype=torch.int8)
    ws=(torch.rand((1,n),generator=g,dtype=torch.float32)*0.02+0.002).to(torch.float16)
    return xq.to("xpu"),xs.to("xpu"),wq.to("xpu"),ws.to("xpu")
bias=torch.Tensor()
def run(m,k,n):
    xq,xs,wq,ws=mk(m,k,n,seed=4242+k+n)
    def call(): return torch.ops._xpu_C.int8_gemm_w8a8(xq,xs,None,wq,ws,None,bias,torch.float16)
    out=call(); torch.xpu.synchronize(); o=out.float()
    fp=f"sum={o.sum().item():+.6e} absmax={o.abs().max().item():.6e}"
    for _ in range(WARMUP): call()
    torch.xpu.synchronize(); t0=time.perf_counter()
    for _ in range(ITERS): call()
    torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/ITERS
    if m==1:
        gbps=(k*n+m*k+m*n*2)/dt/1e9
        print(f"DECODE  k={k:<6} n={n:<6} | {fp} | {dt*1e3:8.4f} ms {gbps:6.1f} GB/s ({100*gbps/BW:4.1f}%)",flush=True)
    else:
        tf=2*m*k*n/dt/1e12
        print(f"PREFILL m={m:<4} k={k:<6} n={n:<6} | {fp} | {dt*1e3:8.4f} ms {tf:7.1f} TFLOP/s ({100*tf/TOPS:4.1f}%)",flush=True)
# prefill FIRST (the headline pp target) on the safe k=5120 shape, then safe decodes,
# then the k=17408 DEVICE_LOST crasher LAST (isolated) so the safe results always land.
for (k,n) in [(5120,17408)]: run(512,k,n)
for (k,n) in [(5120,17408)]: run(2048,k,n)
for (k,n) in [(4096,11008),(5120,17408),(5120,5120)]: run(1,k,n)
print("--- now the k=17408 shapes (may DEVICE_LOST) ---",flush=True)
for (k,n) in [(17408,5120)]: run(512,k,n)
for (k,n) in [(17408,5120)]: run(2048,k,n)
for (k,n) in [(17408,5120)]: run(1,k,n)
print("DONE",flush=True)
