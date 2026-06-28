#!/usr/bin/env python3
# W4A8 ABI gate: can the torch-2.11-built vLLM _xpu_C.abi3.so (with int4_gemm_w4a8)
# LOAD + RUN inside the sglang torch-2.12 image? If yes, the port is a drop-in.
import os, time, sys, torch
DEV="xpu"
KDIR="/work/_v0230_kernels/vllm_xpu_kernels"
SO=KDIR+"/_xpu_C.abi3.so"
print("torch", torch.__version__)
print("loading", SO)
try:
    torch.ops.load_library(SO)
    print("LOAD OK")
except Exception as e:
    print("LOAD FAILED:", repr(e)[:400]); sys.exit(1)
has = hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")
print("int4_gemm_w4a8 registered:", has)
if not has: sys.exit(1)

# real sqgptq down_proj layer (already in the op's packed layout)
import safetensors.torch as st
CKPT="/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
PFX="model.language_model.layers.20.mlp.down_proj"
t=st.load_file(CKPT)
wq=t[f"{PFX}.weight"].to(DEV)          # [N, K/8] int32 packed
ws=t[f"{PFX}.weight_scale"].to(DEV)    # [N, K/g] bf16
N, K8 = wq.shape; K=K8*8; G=128
print(f"layer N={N} K={K} group={G}  wq={tuple(wq.shape)}{wq.dtype} ws={tuple(ws.shape)}{ws.dtype}")
qweight = wq.t().contiguous()          # [K/8, N]
wscale  = ws.t().contiguous()          # [K/g, N]
wzp = torch.tensor([8], dtype=torch.int8, device=DEV)

def sync(): torch.xpu.synchronize()
def bench(fn,iters=60,warm=20):
    for _ in range(warm): fn()
    sync();s=time.time()
    for _ in range(iters): fn()
    sync();return (time.time()-s)/iters*1000.0

def act_q(x):  # per-token sym int8 -> (int8, fp16 scale, int32 zero)
    amax=x.abs().amax(-1,keepdim=True).clamp_(min=1e-5); xs=(amax/127.0).to(x.dtype)
    xq=(x/xs).round().clamp_(-127,127).to(torch.int8)
    zz=torch.zeros_like(amax,dtype=torch.int32)
    return xq.contiguous(), xs.contiguous(), zz.contiguous()

def w4a8(x):
    xq,xs,zz=act_q(x)
    return torch.ops._xpu_C.int4_gemm_w4a8(xq, xs, zz, qweight, wscale, wzp, G, None, None)

print("\n==== run + bench (warm) ====")
for M in (1,2048):
    x=torch.randn(M,K,device=DEV,dtype=torch.float16)*0.1
    try:
        y=w4a8(x)
        tb=bench(lambda: x@torch.zeros(K,N,device=DEV,dtype=torch.float16))  # bf16-ish ref shape (fp16)
        tw=bench(w4a8)
        print(f"M={M:>4} int4_gemm_w4a8={tw:.4f}ms  fp16-matmul={tb:.4f}ms  ({tb/tw:.2f}x)  out={tuple(y.shape)}{y.dtype} finite={torch.isfinite(y).all().item()}")
    except Exception as e:
        print(f"M={M:>4} CALL FAILED:", repr(e)[:300])
print("\nABI GATE: if LOAD OK + op runs finite at M=2048 -> drop-in port viable in sglang image.")
