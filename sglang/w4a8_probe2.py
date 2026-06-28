#!/usr/bin/env python3
# W4A8 gate-2: (a) does torch.compile FUSION realize the int8-XMX prefill win
#   (PERF.md claims 1.6x vs bf16)?  (b) is a FUSED int4xint8 op available on B70
#   (torch.ops._xpu_C.int4_gemm_w4a8 from vLLM / any aten int4 path)?
import os, time, torch
DEV="xpu"
IN, OUT = 17408, 5120
def sync(): torch.xpu.synchronize()
def bench(fn, iters=50, warm=15):
    for _ in range(warm): fn()
    sync(); t=time.time()
    for _ in range(iters): fn()
    sync(); return (time.time()-t)/iters*1000.0

print("torch", torch.__version__)
# ---- (b) probe available fused int-GEMM ops ----
print("\n==== fused-op availability ====")
for mod in ("_xpu_C","_C","torch_ipex","intel_extension_for_pytorch"):
    try:
        m=getattr(torch.ops, mod, None)
        print(f"torch.ops.{mod}:", "present" if m is not None else "absent",
              ([x for x in dir(m) if 'int4' in x.lower() or 'w4a8' in x.lower() or 'woq' in x.lower()][:8] if m is not None else ""))
    except Exception as e: print(f"torch.ops.{mod}: ERR {e}")
for name in ("int4_gemm_w4a8","int4_gemm","woq_gemm"):
    try:
        op=getattr(torch.ops._xpu_C, name, None)
        print(f"  _xpu_C.{name}:", "FOUND" if op is not None else "no")
    except Exception: print(f"  _xpu_C.{name}: no (_xpu_C absent)")
try:
    import auto_round_kernel as ark
    print("auto_round_kernel dir int8/s8:", [x for x in dir(ark) if 's8' in x.lower() or 'int8' in x.lower()][:10])
except Exception as e: print("ark import:", e)

# ---- (a) torch.compile fusion of W8A8 per-channel ----
torch.manual_seed(0)
W=(torch.randn(OUT,IN,device=DEV,dtype=torch.bfloat16)*0.02)
s_ch=(W.abs().amax(1,keepdim=True).clamp_(min=1e-6)/127.0)
Wq8_t=(W/s_ch).round().clamp_(-127,127).to(torch.int8).t().contiguous()  # [in,out]
wscale=s_ch.reshape(1,-1).float()

def w8a8_eager(x):
    amax=x.abs().amax(-1,keepdim=True).clamp_(min=1e-5); xs=amax/127.0
    xq=(x/xs).round().clamp_(-127,127).to(torch.int8)
    return torch._int_mm(xq,Wq8_t).float()*xs.float()*wscale

w8a8_compiled=None
try:
    w8a8_compiled=torch.compile(w8a8_eager, dynamic=False)
except Exception as e: print("compile setup:", e)

print("\n==== W8A8 fusion timings (ms) ====")
print(f"{'M':>6}{'bf16':>10}{'w8a8_eager':>14}{'w8a8_compiled':>16}{'compiled x bf16':>18}")
for M in (1,512,2048):
    x=torch.randn(M,IN,device=DEV,dtype=torch.bfloat16)*0.1
    tb=bench(lambda:x@W.t())
    te=bench(lambda:w8a8_eager(x))
    tc=None
    if w8a8_compiled is not None:
        try:
            _=w8a8_compiled(x); _=w8a8_compiled(x)  # trigger compile
            tc=bench(lambda:w8a8_compiled(x))
        except Exception as e: print(f"  M={M} compiled FAILED:", repr(e)[:120])
    sc=f"{tb/tc:.2f}x" if tc else "N/A"
    fc=f"{tc:.4f}" if tc else "N/A"
    print(f"{M:>6}{tb:>10.4f}{te:>14.4f}{fc:>16}{sc:>18}")
print("\nGATE-2: if w8a8_compiled M>=512 >= ~1.3x bf16 -> int8-XMX prefill win is REAL")
print("        and a W4A8 prefill that reaches it (via fused int4->int8) is worth building.")
