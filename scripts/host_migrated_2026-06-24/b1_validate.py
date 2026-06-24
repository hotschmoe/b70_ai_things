import os, time, torch
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.*
MODE=os.environ.get("MODE","?"); ITERS=int(os.environ.get("ITERS","200")); WARMUP=int(os.environ.get("WARMUP","30"))
BW=608.0
print(f"MODE={MODE} torch={torch.__version__} has_int4={hasattr(torch.ops._xpu_C,'int4_gemm_w4a8')}", flush=True)

def make_inputs(m,k,n,seed):
    g=torch.Generator().manual_seed(seed)
    # symmetric per-token int8 activations (zero src-zp -- the production path)
    xq=torch.randint(-127,128,(m,k),generator=g,dtype=torch.int8)
    xs=(torch.rand((m,1),generator=g,dtype=torch.float32)*0.02+0.005).to(torch.float16)
    xzp=torch.zeros((m,1),dtype=torch.int32)
    # int4 weights packed 8-per-int32 -> [k/8, n], symmetric weight zp=8
    wb=torch.randint(-128,128,((k*n)//2,),generator=g,dtype=torch.int8).view(torch.int32).reshape(k//8,n)
    gsz=min(128,k); gn=k//gsz
    ws=(torch.rand((gn,n),generator=g,dtype=torch.float32)*0.02+0.002).to(torch.float16)
    wzp=torch.tensor([8],dtype=torch.int8)
    return (xq.to("xpu"), xs.to("xpu"), xzp.to("xpu"),
            wb.to("xpu").transpose(0,1).contiguous().transpose(0,1),
            ws.to("xpu"), wzp.to("xpu"), gsz)

SHAPES=[(1,4096,11008),(1,5120,17408),(1,17408,5120),(1,5120,5120)]
bias=torch.Tensor()
for (m,k,n) in SHAPES:
    xq,xs,xzp,wb,ws,wzp,gsz=make_inputs(m,k,n,seed=1234+k+n)
    def call(): return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,xzp,wb,ws,wzp,gsz,None,bias)
    out=call(); torch.xpu.synchronize()
    # correctness fingerprint (deterministic inputs -> compare across MODE)
    o=out.float()
    fp_sum=o.sum().item(); fp_absmax=o.abs().max().item(); fp_mean=o.mean().item()
    for _ in range(WARMUP): call()
    torch.xpu.synchronize(); t0=time.perf_counter()
    for _ in range(ITERS): call()
    torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/ITERS
    gbps=(k*n*0.5 + m*k + m*n*2)/dt/1e9
    print(f"SHAPE k={k:<6} n={n:<6} | sum={fp_sum:+.6e} absmax={fp_absmax:.6e} mean={fp_mean:+.6e} "
          f"| {dt*1e3:8.4f} ms {gbps:6.1f} GB/s ({100*gbps/BW:4.1f}%)", flush=True)
print("DONE", flush=True)
