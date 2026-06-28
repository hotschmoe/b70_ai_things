#!/usr/bin/env python3
# Test the pip wheel vllm-xpu-kernels==0.1.3.1 (torch-2.12 era) in the sglang image:
# does its int4_gemm_w4a8 load (ABI match) + run + stay fast on B70?
import os, sys, time, glob, torch
DEV="xpu"
print("torch", torch.__version__)
import vllm_xpu_kernels
kdir=os.path.dirname(vllm_xpu_kernels.__file__)
print("pkg dir:", kdir)
print("contents:", sorted(os.listdir(kdir))[:30])
so=glob.glob(kdir+"/_xpu_C*.so")
print("xpu_C so:", so)
loaded=False
# Strategy 1: proper extension import (runs TORCH_LIBRARY registration)
try:
    import vllm_xpu_kernels._xpu_C  # noqa
    loaded=True; print("LOAD OK via import vllm_xpu_kernels._xpu_C")
except Exception as e:
    print("import _xpu_C failed:", repr(e)[:200])
# Strategy 2: ctypes RTLD_GLOBAL (proven to work; static init registers ops)
if not loaded and so:
    import ctypes
    try:
        ctypes.CDLL(so[0], mode=ctypes.RTLD_GLOBAL); loaded=True
        print("LOAD OK via ctypes RTLD_GLOBAL")
    except Exception as e:
        print("ctypes load failed:", repr(e)[:300])
has=hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")
print("int4_gemm_w4a8 registered:", has)
if not has:
    print("ops available:", [x for x in dir(torch.ops._xpu_C) if not x.startswith('_')][:40]); sys.exit(1)

import safetensors.torch as st
CKPT="/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
PFX="model.language_model.layers.20.mlp.down_proj"
t=st.load_file(CKPT)
wq=t[f"{PFX}.weight"].to(DEV); ws=t[f"{PFX}.weight_scale"].to(DEV)
N,K8=wq.shape; K=K8*8; G=128
qweight=wq.t().contiguous(); wscale=ws.t().contiguous(); wzp=torch.tensor([8],dtype=torch.int8,device=DEV)
print(f"N={N} K={K} qweight={tuple(qweight.shape)} wscale={tuple(wscale.shape)}")
def sync(): torch.xpu.synchronize()
def bench(fn,iters=60,warm=20):
    for _ in range(warm): fn()
    sync();s=time.time()
    for _ in range(iters): fn()
    sync();return (time.time()-s)/iters*1000.0
def act_q(x):
    amax=x.abs().amax(-1,keepdim=True).clamp_(min=1e-5);xs=(amax/127.0).to(x.dtype)
    return (x/xs).round().clamp_(-127,127).to(torch.int8).contiguous(), xs.contiguous(), torch.zeros_like(amax,dtype=torch.int32).contiguous()
def w4a8(x):
    xq,xs,zz=act_q(x); return torch.ops._xpu_C.int4_gemm_w4a8(xq,xs,zz,qweight,wscale,wzp,G,None,None)
print("\n==== run + bench ====")
for M in (1,2048):
    x=torch.randn(M,K,device=DEV,dtype=torch.float16)*0.1
    try:
        y=w4a8(x)
        tb=bench(lambda: x@torch.zeros(K,N,device=DEV,dtype=torch.float16))
        tw=bench(w4a8)
        print(f"M={M:>4} int4_gemm_w4a8={tw:.4f}ms fp16mm={tb:.4f}ms ({tb/tw:.2f}x) out={tuple(y.shape)}{y.dtype} finite={torch.isfinite(y).all().item()}")
    except Exception as e:
        print(f"M={M:>4} CALL FAILED:", repr(e)[:300])
print("\nWHEEL GATE: LOAD OK + finite M=2048 -> install vllm-xpu-kernels==0.1.3.1 into sglang image = the port.")
