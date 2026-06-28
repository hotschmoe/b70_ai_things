#!/usr/bin/env python3
# W4A8 decision-gate microbench (sglang-xpu:woq image, card 0).
# Compares, on a REAL Lorbus int4 down_proj layer (in=17408, out=5120):
#   bf16 matmul | woqgemm int4 | W8A8 per-channel _int_mm | W4A8 grouped-128 _int_mm
# at M=1 (decode) and M=2048 (prefill), INCLUDING honest int4->int8 conversion cost.
# GATE: does an int8-XMX prefill path HANDILY beat bf16 (=> >2x int4-woq's 0.56x)
#       while decode (woqgemm) stays at the int4-champion level?
import os, time, torch
import safetensors.torch as st

DEV = "xpu"
CKPT = "/models/Lorbus_Qwen3.6-27B-int4-AutoRound"
SHARD = CKPT + "/model-00002-of-00010.safetensors"
PFX = "model.language_model.layers.20.mlp.down_proj"
GS = 128

def sync(): torch.xpu.synchronize()

def bench(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    sync(); t=time.time()
    for _ in range(iters): fn()
    sync(); return (time.time()-t)/iters*1000.0  # ms

def main():
    print("torch", torch.__version__, "xpu", torch.xpu.is_available())
    # --- load real int4 layer ---
    keys=[f"{PFX}.qweight", f"{PFX}.qzeros", f"{PFX}.scales"]
    t = st.load_file(SHARD)
    qweight = t[keys[0]].to(DEV)          # [in//8, out] int32
    qzeros  = t[keys[1]].to(DEV)          # [in//g, out//8] int32
    scales  = t[keys[2]].to(DEV)          # [in//g, out] (fp16/bf16)
    IN  = qweight.shape[0]*8
    OUT = qweight.shape[1]
    print(f"layer down_proj IN={IN} OUT={OUT} qweight={tuple(qweight.shape)} "
          f"scales={tuple(scales.shape)} groups={IN//GS}")

    # --- build woqgemm int4 kernel (faithful) ---
    woq_ok=False
    try:
        from auto_round_kernel.qlinear import QuantLinearGPTQ
        ql = QuantLinearGPTQ(4, GS, True, IN, OUT, False, torch.bfloat16).to(DEV)
        ql.qweight.data = qweight
        ql.qzeros.data  = qzeros
        ql.scales.data  = scales.to(torch.float16)
        ql.post_init()
        woq_ok=True
    except Exception as e:
        print("woqgemm build FAILED:", repr(e))

    # --- realistic bf16 weight + int8 derivations (shape-faithful, for timing) ---
    torch.manual_seed(0)
    W = (torch.randn(OUT, IN, device=DEV, dtype=torch.bfloat16) * 0.02)  # [out,in]
    # per-channel int8 (group_size=-1): single _int_mm path (best-case int8)
    s_ch = (W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6) / 127.0)   # [out,1]
    Wq8_ch = (W / s_ch).round().clamp_(-127,127).to(torch.int8)          # [out,in]
    Wq8_ch_t = Wq8_ch.t().contiguous()                                   # [in,out] int8
    wscale_ch = s_ch.reshape(1,-1).float()                               # [1,out]
    # grouped-128 int8: weight int8 + per-(out,group) scale
    Wg = W.reshape(OUT, IN//GS, GS)
    s_g = (Wg.abs().amax(dim=2).clamp_(min=1e-6) / 127.0)                # [out, ngrp]
    Wq8_g = (Wg / s_g.unsqueeze(2)).round().clamp_(-127,127).to(torch.int8).reshape(OUT, IN)
    Wq8_g_t = Wq8_g.t().contiguous()                                     # [in,out] int8
    s_g_f = s_g.float()                                                  # [out,ngrp]

    def act_q(x):  # per-token sym int8
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
        xs = amax/127.0
        xq = (x/xs).round().clamp_(-127,127).to(torch.int8)
        return xq, xs

    results={}
    for M in (1, 2048):
        x = torch.randn(M, IN, device=DEV, dtype=torch.bfloat16)*0.1
        # bf16 baseline
        def f_bf16(): return x @ W.t()
        results[("bf16",M)] = bench(f_bf16)
        # woqgemm
        if woq_ok:
            try:
                _=ql(x); results[("woq_int4",M)] = bench(lambda: ql(x))
            except Exception as e:
                print(f"woq M={M} FAILED:", repr(e)); results[("woq_int4",M)]=None
        # W8A8 per-channel single _int_mm (incl act quant + dequant)
        def f_w8a8():
            xq,xs = act_q(x)
            acc = torch._int_mm(xq, Wq8_ch_t)               # [M,out] int32
            return acc.float()*xs.float()*wscale_ch
        try:
            _=f_w8a8(); results[("w8a8_perchan",M)] = bench(f_w8a8)
        except Exception as e:
            print(f"w8a8 M={M} FAILED:", repr(e)); results[("w8a8_perchan",M)]=None
        # W4A8 grouped-128 _int_mm loop (incl act quant + per-group accumulate)
        def f_w4a8_grouped():
            xq,xs = act_q(x)
            acc = torch.zeros(M, OUT, device=DEV, dtype=torch.float32)
            for g in range(IN//GS):
                a = xq[:, g*GS:(g+1)*GS].contiguous()
                w = Wq8_g_t[g*GS:(g+1)*GS, :]
                acc += torch._int_mm(a, w).float() * s_g_f[:, g]
            return acc*xs.float()
        try:
            _=f_w4a8_grouped(); results[("w4a8_grouped",M)] = bench(f_w4a8_grouped, iters=20, warm=4)
        except Exception as e:
            print(f"w4a8_grouped M={M} FAILED:", repr(e)); results[("w4a8_grouped",M)]=None

    # int4 -> int8 nibble unpack microcost (the extra conversion for the grouped path)
    def f_unpack():
        qw = qweight.view(torch.int32)
        out = torch.empty(IN, OUT, device=DEV, dtype=torch.int8)
        for j in range(8):
            out[j::8, :] = ((qw >> (4*j)) & 0xF).to(torch.int8) - 8
        return out
    try:
        _=f_unpack(); unpack_ms = bench(f_unpack, iters=20, warm=4)
    except Exception as e:
        print("unpack FAILED:", repr(e)); unpack_ms=None

    # correctness sanity (M=8): int8 candidates vs their own bf16 ref
    x = torch.randn(8, IN, device=DEV, dtype=torch.bfloat16)*0.1
    ref = (x @ W.t()).float()
    def relerr(y): return (y-ref).norm().item()/ref.norm().item()
    xq,xs = act_q(x)
    y8 = (torch._int_mm(xq, Wq8_ch_t).float()*xs.float()*wscale_ch)
    accg = torch.zeros(8,OUT,device=DEV,dtype=torch.float32)
    for g in range(IN//GS):
        accg += torch._int_mm(xq[:,g*GS:(g+1)*GS], Wq8_g_t[g*GS:(g+1)*GS,:]).float()*s_g_f[:,g]
    yg = accg*xs.float()

    print("\n==== TIMINGS (ms/call) ====")
    print(f"{'candidate':<18}{'M=1':>12}{'M=2048':>12}")
    for name in ("bf16","woq_int4","w8a8_perchan","w4a8_grouped"):
        r1=results.get((name,1)); r2=results.get((name,2048))
        f1 = f"{r1:.4f}" if r1 else "  N/A"
        f2 = f"{r2:.4f}" if r2 else "  N/A"
        print(f"{name:<18}{f1:>12}{f2:>12}")
    print(f"\nint4->int8 unpack microcost: {unpack_ms:.4f} ms" if unpack_ms else "unpack N/A")
    print("\n==== SPEEDUP vs bf16 (x>1 = faster) ====")
    for M in (1,2048):
        b=results.get(("bf16",M))
        line=f"M={M}: "
        for name in ("woq_int4","w8a8_perchan","w4a8_grouped"):
            r=results.get((name,M))
            line += f"{name}={b/r:.2f}x  " if (r and b) else f"{name}=N/A  "
        print(line)
    print(f"\ncorrectness rel-err vs bf16(ref): w8a8_perchan={relerr(y8):.4f}  w4a8_grouped={relerr(yg):.4f}")
    print("\nGATE: w4a8 prefill (M=2048) must beat bf16 (>1x) AND woq decode (M=1) stays ~2x bf16.")

if __name__=="__main__":
    main()
