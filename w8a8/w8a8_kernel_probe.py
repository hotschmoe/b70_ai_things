#!/usr/bin/env python3
# w8a8_kernel_probe.py -- W8A8 kernel-envelope microbench for Qwen3.6-27B on B70 (card 0).
#
# Answers the three questions that gate the whole sglang-W8A8 campaign:
#   Q1 (the #1 lever): does XPUGraph CAPTURE recover the M=1 decode launch penalty?
#        The shipped sglang W8A8 decode is EAGER: per layer it fires 3 kernels
#        (per-token int8 act-quant -> torch._int_mm -> fp32 dequant). At M=1 the GEMM
#        itself is ~1.9x bf16 but the chain is launch-bound. Graph capture removed the
#        same penalty for W4A8 (9.4 -> 27 t/s). We capture the chain here in isolation.
#   Q2 (baseline): how fast are the fp8_gemm ops (w8a16 decode, w8a8 prefill) on B70?
#        B70 has NO native FP8 -> oneDNN emulates. This is the bar W8A8 must "handily beat".
#   Q3 (prefill): confirm torch._int_mm int8-XMX prefill (M=2048) speedup vs bf16.
#
# All on REAL Qwen3.6-27B linear shapes, synthetic weights (timing is value-independent;
# we still check finiteness + relerr where a reference exists). Pure card-0 microbench:
# NO serve, NO TP=2, NO wedge risk.
#
# Run: see w8a8/run_kernel_probe.sh
import os, sys, time, ctypes, traceback
import torch

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/w4a8_kernel/_xpu_C.abi3.so")

print("torch", torch.__version__, "xpu_avail", torch.xpu.is_available(), flush=True)
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL RTLD_GLOBAL OK:", SO, flush=True)
    HAVE_SO = True
except OSError as e:
    print("CDLL FAILED (fp8 baselines will be skipped):", str(e)[:200], flush=True)
    HAVE_SO = False

HAVE_FP8_W8A16 = HAVE_SO and hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")
HAVE_FP8_W8A8 = HAVE_SO and hasattr(torch.ops._xpu_C, "fp8_gemm_w8a8")
print(f"ops: fp8_gemm_w8a16={HAVE_FP8_W8A16} fp8_gemm_w8a8={HAVE_FP8_W8A8} "
      f"_int_mm=builtin", flush=True)

# Real Qwen3.6-27B full (TP=1) linear shapes [N=out, K=in]
SHAPES = [
    ("gate_up_proj", 34816, 5120),   # biggest single GEMM (N*K weight bytes)
    ("down_proj",     5120, 17408),  # the big contract dim
    ("qkv_full_attn", 14336, 5120),
]
MS = [1, 2048]   # decode, prefill


def sync():
    torch.xpu.synchronize()


def bench(fn, warm=30, iters=80):
    for _ in range(warm):
        fn()
    sync(); s = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - s) / iters * 1000.0


def make_int8_weight(N, K):
    # synthetic per-channel int8 weight; weight_t [K,N] int8 (what torch._int_mm wants), wscale [1,N]
    W = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02)
    amax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)          # [N,1]
    wscale = (amax / 127.0)                                            # [N,1]
    Wq = torch.round(W / wscale).clamp_(-127, 127).to(torch.int8)      # [N,K]
    weight_t = Wq.t().contiguous()                                     # [K,N] int8
    wscale_row = wscale.reshape(1, N).to(torch.float32)                # [1,N]
    return W, weight_t, wscale_row


def int8_chain(x, weight_t, wscale_row):
    # the EXACT shipped sglang w8a8_shim chain: act-quant -> _int_mm -> dequant
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    x_scale = amax * (1.0 / 127.0)
    x_q = torch.round(x / x_scale).clamp_(-127, 127).to(torch.int8)
    acc = torch._int_mm(x_q, weight_t)                                 # [M,N] int32
    out = acc.to(torch.float32) * x_scale.to(torch.float32) * wscale_row
    return out.to(torch.float16)


def run_shape(name, N, K):
    print(f"\n=================  {name}  [N={N} K={K}]  weight={N*K/1e6:.0f}M params  "
          f"({N*K/1e9:.2f}GB int8)  =================", flush=True)
    W, weight_t, wscale_row = make_int8_weight(N, K)
    w_fp16 = W.to(torch.float16)                                       # [N,K] bf16->fp16 baseline weight

    # fp8 weight (e4m3), NT layout B=[K,N] view, per-channel scale
    fp8_B = None; fp8_Bscale = None
    if HAVE_FP8_W8A16:
        try:
            wamax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)     # [N,1]
            fp8_scale = (wamax / 448.0)                                    # e4m3 max ~448
            Wf8 = (W / fp8_scale).clamp_(-448, 448).to(torch.float8_e4m3fn)  # [N,K]
            fp8_B = Wf8.t()                                                # [K,N] NT view
            fp8_Bscale = fp8_scale.reshape(N).to(torch.float16)           # [N]
        except Exception as e:
            print("  fp8 weight prep failed:", repr(e)[:120], flush=True)

    for M in MS:
        tag = "DECODE" if M == 1 else f"PREFILL M={M}"
        x = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
        print(f"  --- {tag} ---", flush=True)

        # bf16 reference
        try:
            tb = bench(lambda: x @ w_fp16.t())
        except Exception as e:
            print("    bf16 ref FAILED:", repr(e)[:120], flush=True); continue
        print(f"    bf16 matmul            {tb:8.4f} ms   1.00x  (baseline)", flush=True)

        # int8 GEMM-only (isolates DPAS from the chain overhead)
        xq = torch.round(x / (x.abs().amax(-1, keepdim=True) * (1/127.))).clamp_(-127, 127).to(torch.int8)
        try:
            tg = bench(lambda: torch._int_mm(xq, weight_t))
            print(f"    int8 _int_mm GEMM-only {tg:8.4f} ms  {tb/tg:5.2f}x  (DPAS, no quant/dequant)", flush=True)
        except Exception as e:
            print("    int8 GEMM-only FAILED:", repr(e)[:120], flush=True)

        # int8 full chain EAGER
        try:
            y_eager = int8_chain(x, weight_t, wscale_row)
            fin = torch.isfinite(y_eager).all().item()
            ref = (x.to(torch.float32) @ w_fp16.t().to(torch.float32))
            relerr = ((y_eager.to(torch.float32) - ref).norm() / ref.norm()).item()
            te = bench(lambda: int8_chain(x, weight_t, wscale_row))
            print(f"    int8 chain EAGER       {te:8.4f} ms  {tb/te:5.2f}x  finite={fin} relerr={relerr:.2e}", flush=True)
        except Exception as e:
            print("    int8 chain EAGER FAILED:", repr(e)[:160], flush=True); te = None

        # int8 full chain XPUGraph-CAPTURED  (THE Q1 lever)
        try:
            x_static = x.clone()
            # warmup (oneDNN primitive creation / allocator outside capture)
            for _ in range(10):
                _ = int8_chain(x_static, weight_t, wscale_row)
            sync()
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                out_static = int8_chain(x_static, weight_t, wscale_row)
            sync()
            # validate replay correctness
            x_static.copy_(x)
            g.replay(); sync()
            fin_c = torch.isfinite(out_static).all().item()
            relc = ((out_static.to(torch.float32) - ref).norm() / ref.norm()).item()
            tc = bench(lambda: g.replay())
            spd = tb / tc
            vs_eager = (te / tc) if te else float('nan')
            print(f"    int8 chain GRAPH       {tc:8.4f} ms  {spd:5.2f}x  finite={fin_c} relerr={relc:.2e}"
                  f"   [vs eager-chain {vs_eager:.2f}x]  <== Q1", flush=True)
        except Exception as e:
            print("    int8 chain GRAPH FAILED:", repr(e)[:200], flush=True)
            if os.environ.get("PROBE_TB"): traceback.print_exc()

        # fp8 w8a16 (fp16 act x fp8 weight) -- the FP8 baseline to beat
        if HAVE_FP8_W8A16 and fp8_B is not None:
            ok = False
            for sc_desc, sc in [("scale[N]", fp8_Bscale), ("scale[N,1]", fp8_Bscale.reshape(N, 1)),
                                ("scale[1,N]", fp8_Bscale.reshape(1, N)), ("None", None)]:
                try:
                    y = torch.ops._xpu_C.fp8_gemm_w8a16(x, fp8_B, sc, None)
                    if not torch.isfinite(y).all().item():
                        continue
                    tf = bench(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(x, fp8_B, sc, None))
                    print(f"    fp8 w8a16 ({sc_desc:9s})  {tf:8.4f} ms  {tb/tf:5.2f}x  <== FP8 baseline", flush=True)
                    ok = True; break
                except Exception as e:
                    last = repr(e)[:120]
            if not ok:
                print(f"    fp8 w8a16 FAILED (all scale layouts): {last}", flush=True)

        # fp8 w8a8 (fp8 act x fp8 weight) -- prefill FP8 baseline
        if HAVE_FP8_W8A8 and fp8_B is not None and M > 1:
            try:
                xamax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-8)
                xscale = (xamax / 448.0)
                xf8 = (x / xscale).clamp_(-448, 448).to(torch.float8_e4m3fn)
                xscale_t = xscale.reshape(M).to(torch.float16)
                ok = False
                for desc, args in [("A,As,B,Bs,None", (xf8, xscale_t, fp8_B, fp8_Bscale, None)),
                                   ("A,As,B,Bs", (xf8, xscale_t, fp8_B, fp8_Bscale))]:
                    try:
                        y = torch.ops._xpu_C.fp8_gemm_w8a8(*args)
                        tf = bench(lambda: torch.ops._xpu_C.fp8_gemm_w8a8(*args))
                        print(f"    fp8 w8a8 ({desc:14s}) {tf:8.4f} ms  {tb/tf:5.2f}x  finite={torch.isfinite(y).all().item()}", flush=True)
                        ok = True; break
                    except Exception as e:
                        last = repr(e)[:120]
                if not ok:
                    print(f"    fp8 w8a8 FAILED: {last}", flush=True)
            except Exception as e:
                print("    fp8 w8a8 setup FAILED:", repr(e)[:120], flush=True)


def main():
    for (name, N, K) in SHAPES:
        run_shape(name, N, K)
    print("\nGATES: Q1 int8 chain GRAPH should beat int8 chain EAGER at M=1 (launch penalty recovered).", flush=True)
    print("       Q2 int8 chain GRAPH (M=1) and int8 _int_mm (M=2048) should beat fp8 ops (W8A8 > FP8).", flush=True)


if __name__ == "__main__":
    main()
