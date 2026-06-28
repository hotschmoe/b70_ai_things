#!/usr/bin/env python3
# W4A8 gate-4: (1) woqgemm(compute_type=int8) on REAL int4 weight = the fused W4A8 kernel
#   -- does it run / fall back to fp16 / correct / fast on this box (oneAPI<2026)?
# (2) woqgemm_s8 fused int8 (single-launch) timing at M=1/M=2048 vs naive _int_mm.
import time, torch
import auto_round_kernel as ark
from auto_round_kernel.qlinear import QuantLinearGPTQ
import safetensors.torch as st
DEV="xpu"; GS=128
CKPT="/models/Lorbus_Qwen3.6-27B-int4-AutoRound"
SHARD=CKPT+"/model-00002-of-00010.safetensors"; PFX="model.language_model.layers.20.mlp.down_proj"
def sync(): torch.xpu.synchronize()
def bench(fn,iters=50,warm=15):
    for _ in range(warm): fn()
    sync();t=time.time()
    for _ in range(iters): fn()
    sync();return (time.time()-t)/iters*1000.0

# cvtstr_dtype accepted strings
try:
    import inspect
    print("cvtstr_dtype src:\n", inspect.getsource(ark.cvtstr_dtype)[:800])
except Exception as e: print("cvtstr:",e)

t=st.load_file(SHARD)
qw=t[f"{PFX}.qweight"].to(DEV); qz=t[f"{PFX}.qzeros"].to(DEV); sc=t[f"{PFX}.scales"].to(DEV)
IN=qw.shape[0]*8; OUT=qw.shape[1]
ql=QuantLinearGPTQ(4,GS,True,IN,OUT,False,torch.bfloat16).to(DEV)
ql.qweight.data=qw; ql.qzeros.data=qz; ql.scales.data=sc.to(torch.float16); ql.post_init()
print(f"\nql cdt={ql.cdt!r} wdt={ql.wdt!r} sdt={ql.sdt!r} asym={ql.asym} gs={ql.group_size} "
      f"packed qweight={tuple(ql.qweight.shape)} {ql.qweight.dtype}")
bias=ql.bias

print("\n==== (1) woqgemm compute_type sweep (real int4 weight) ====")
for M in (1,2048):
    x=torch.randn(M,IN,device=DEV,dtype=torch.bfloat16)*0.1
    ref=ql(x).float()  # current fp16-compute path
    for ct in (ql.cdt, "int8", "int8_fp32", "bf16"):
        try:
            y=ark.woqgemm(x.to(ql.torch_dt), ql.qweight, bias, OUT, IN, ql.group_size, ct, ql.wdt, ql.sdt, ql.asym)
            err=(y.float()-ref).norm().item()/ref.norm().item()
            ms=bench(lambda: ark.woqgemm(x.to(ql.torch_dt), ql.qweight, bias, OUT, IN, ql.group_size, ct, ql.wdt, ql.sdt, ql.asym))
            print(f"  M={M:>4} compute_type={ct!r:<12} ms={ms:.4f} relerr={err:.4f}")
        except Exception as e:
            print(f"  M={M:>4} compute_type={ct!r:<12} FAILED: {repr(e)[:90]}")

print("\n==== (2) woqgemm_s8 fused int8 (single launch) vs bf16 vs naive _int_mm ====")
W=(torch.randn(OUT,IN,device=DEV,dtype=torch.bfloat16)*0.02)
s_ch=(W.abs().amax(1,keepdim=True).clamp_(min=1e-6)/127.0)        # [out,1]
Wq8=(W/s_ch).round().clamp_(-127,127).to(torch.int8)             # [out,in]=[n,k]
Wq8_t=Wq8.t().contiguous()                                       # [in,out] for _int_mm
scaleB=s_ch.reshape(-1).float()                                  # [out]
bias0=torch.zeros(OUT,device=DEV,dtype=torch.bfloat16)
def act_q(x):
    amax=x.abs().amax(-1,keepdim=True).clamp_(min=1e-5);xs=amax/127.0
    return (x/xs).round().clamp_(-127,127).to(torch.int8),xs
for M in (1,2048):
    x=torch.randn(M,IN,device=DEV,dtype=torch.bfloat16)*0.1
    tb=bench(lambda:x@W.t())
    # woqgemm_s8: A int8, B int8 [n,k], scaleB, bias
    def f_s8():
        xq,xs=act_q(x)
        return ark.woqgemm_s8(xq, Wq8, scaleB, bias0)  # does it apply act scale? check
    try:
        xq,xs=act_q(x); yraw=ark.woqgemm_s8(xq,Wq8,scaleB,bias0)
        ref=(x@W.t()).float()
        # try interpreting output (may already be dequant by weight scale; we still need act scale)
        cand=yraw.float()*xs.float()
        err=(cand-ref).norm().item()/ref.norm().item()
        ts8=bench(f_s8)
        print(f"  M={M:>4} woqgemm_s8 ms={ts8:.4f} ({tb/ts8:.2f}x bf16)  relerr(after*act)={err:.4f}  rawdtype={yraw.dtype}")
    except Exception as e:
        print(f"  M={M:>4} woqgemm_s8 FAILED: {repr(e)[:110]}")
    # naive _int_mm for ref
    def f_im():
        xq,xs=act_q(x);return torch._int_mm(xq,Wq8_t).float()*xs.float()*s_ch.reshape(1,-1).float()
    try: print(f"        naive _int_mm ms={bench(f_im):.4f} ({tb/bench(f_im):.2f}x)  bf16={tb:.4f}")
    except Exception as e: print("   _int_mm fail",repr(e)[:80])
