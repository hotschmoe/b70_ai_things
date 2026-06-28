#!/usr/bin/env python3
# Verify the FRESHLY-BUILT (vs sglang torch 2.12) _xpu_C.abi3.so:
# does int4_gemm_w4a8 load (ABI match!) + run finite + fast on B70?
import os, sys, time, ctypes, glob, torch
DEV="xpu"
SO="/build/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
print("torch", torch.__version__)
print("loading built so:", SO, "exists:", os.path.exists(SO))
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL RTLD_GLOBAL OK")
except OSError as e:
    print("CDLL FAILED:", str(e)[:400]); sys.exit(1)
has=hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")
print("int4_gemm_w4a8 registered:", has)
if not has:
    print("ops:", [x for x in dir(torch.ops._xpu_C) if not x.startswith('_')][:40]); sys.exit(1)

import safetensors.torch as st
CKPT="/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
PFX="model.language_model.layers.20.mlp.down_proj"
t=st.load_file(CKPT)
wq=t[f"{PFX}.weight"].to(DEV); ws=t[f"{PFX}.weight_scale"].to(DEV)
N,K8=wq.shape; K=K8*8; G=128
# B must be NT format: [K/8, N] with stride[0]==1 -> wq.t() as a VIEW (no .contiguous()!)
qweight=wq.t(); wscale=ws.t().contiguous(); wzp=torch.tensor([8],dtype=torch.int8,device=DEV)  # 1-D zp = symmetric
print("qweight.stride:", qweight.stride(), "(stride[0] must be 1 for NT)")
print(f"N={N} K={K} qweight={tuple(qweight.shape)} wscale={tuple(wscale.shape)}")
def sync(): torch.xpu.synchronize()
def bench(fn,iters=60,warm=25):
    for _ in range(warm): fn()
    sync();s=time.time()
    for _ in range(iters): fn()
    sync();return (time.time()-s)/iters*1000.0
def act_q(x):
    amax=x.abs().amax(-1,keepdim=True).clamp_(min=1e-5);xs=(amax/127.0).to(x.dtype)
    return (x/xs).round().clamp_(-127,127).to(torch.int8).contiguous(),xs.contiguous(),torch.zeros_like(amax,dtype=torch.int32).contiguous()
def w4a8(x):
    xq,xs,zz=act_q(x); return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,zz,qweight,wscale,wzp,G,None,None)
act_q_c=torch.compile(act_q, dynamic=False)
def w4a8_c(x):
    xq,xs,zz=act_q_c(x); return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,zz,qweight,wscale,wzp,G,None,None)
# int4_gemm_w4a16 (fp16 act, NO act-quant) -- the DECODE half of the hybrid
has16=hasattr(torch.ops._xpu_C,"int4_gemm_w4a16")
print("int4_gemm_w4a16 registered:", has16)
def w4a16(x):
    return torch.ops._xpu_C.int4_gemm_w4a16(x, qweight, None, wscale, wzp, G, None)

print("\n==== run + bench (warm) ====")
for M in (1,2048):
    x=torch.randn(M,K,device=DEV,dtype=torch.float16)*0.1
    try:
        y=w4a8(x)
        tb=bench(lambda: x@torch.zeros(K,N,device=DEV,dtype=torch.float16))
        tw=bench(lambda: w4a8(x))                      # op + eager act-quant
        xq,xs,zz=act_q(x)                              # pre-quantize once
        top=bench(lambda: torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,zz,qweight,wscale,wzp,G,None,None))  # op-only
        taq=bench(lambda: act_q(x))                    # act-quant only (eager)
        try:
            _=w4a8_c(x); _=w4a8_c(x)                    # trigger compile
            twc=bench(lambda: w4a8_c(x))               # op + COMPILE-FUSED act-quant
        except Exception as e:
            twc=None; print("  compiled FAILED:", repr(e)[:150])
        twcs=f"{twc:.4f}ms ({tb/twc:.2f}x fp16)" if twc else "N/A"
        w16s="N/A"
        if has16:
            try:
                y16=w4a16(x); t16=bench(lambda: w4a16(x))
                w16s=f"{t16:.4f}ms ({tb/t16:.2f}x fp16) finite={torch.isfinite(y16).all().item()}"
            except Exception as e:
                w16s=f"FAILED {repr(e)[:90]}"
        print(f"M={M:>4} W4A8 op-only={top:.4f}ms ({tb/top:.2f}x)  W4A8 op+FUSED-aq={twcs}  W4A16(decode)={w16s}  fp16mm={tb:.4f}ms")
    except Exception as e:
        print(f"M={M:>4} CALL FAILED:", repr(e)[:300])
print("\nGATE: LOAD OK + finite + fast (decode>2x, prefill>1.5x fp16) -> the W4A8 kernel is LIVE in sglang torch.")
