#!/usr/bin/env python3
# w8a8_fused_probe.py -- validate the NEW fused int8 oneDNN ops on B70 (card 0).
#
#   int8_gemm_w8a16(A_f16[M,K], B_s8[K,N] NT, B_scale[N], bias?)  -> [M,N]   (DECODE: 1 launch)
#   int8_gemm_w8a8 (A_s8[M,K], A_scale[M,1], A_zp?, B_s8[K,N] NT, B_scale[N], azp?, bias?, out_dtype) (PREFILL: fused scale)
#   dynamic_per_token_int8_quant(x, sym=True, bits=8) -> (q[...,K] s8, scale[...,1], zp[...,1])  (fused act-quant)
#
# Gates: decode int8_gemm_w8a16 should match fp8_gemm_w8a16 (~1.9x bf16) in ONE launch and beat
#        the 3-kernel _int_mm chain (1.3-1.5x captured). Prefill int8_gemm_w8a8 should realize the
#        int8-XMX 1.7-1.9x (vs the un-fused chain 0.7-0.8x). Both validated for finiteness + relerr.
import os, sys, time, ctypes, traceback
import torch

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/kernel/_xpu_C.abi3.so")
print("torch", torch.__version__, "xpu", torch.xpu.is_available(), flush=True)
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
    print("CDLL OK:", SO, flush=True)
except OSError as e:
    print("CDLL FAILED:", str(e)[:300], flush=True); sys.exit(1)

ops = torch.ops._xpu_C
for nm in ["int8_gemm_w8a16", "int8_gemm_w8a8", "fp8_gemm_w8a16", "dynamic_per_token_int8_quant"]:
    print(f"  op {nm}: {hasattr(ops, nm)}", flush=True)
assert hasattr(ops, "int8_gemm_w8a16"), "int8_gemm_w8a16 NOT registered -- build failed to add it"

SHAPES = [("gate_up", 34816, 5120), ("down_proj", 5120, 17408), ("qkv", 14336, 5120)]


def sync(): torch.xpu.synchronize()
def bench(fn, warm=30, iters=80):
    for _ in range(warm): fn()
    sync(); s = time.time()
    for _ in range(iters): fn()
    sync(); return (time.time() - s) / iters * 1000.0


def q_weight_s8(W):
    # W [N,K] f32 -> Wq [N,K] s8 (per-channel sym), wscale [N]
    amax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    wscale = (amax / 127.0)
    Wq = torch.round(W / wscale).clamp_(-127, 127).to(torch.int8)
    return Wq, wscale.reshape(-1)


def call_w8a16(A, B_nt, Bsc):
    return ops.int8_gemm_w8a16(A, B_nt, Bsc, None)


for (name, N, K) in SHAPES:
    print(f"\n===== {name} [N={N} K={K}] =====", flush=True)
    W = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02)
    Wq, wscale = q_weight_s8(W)                       # [N,K] s8, [N]
    B_nt = Wq.t()                                     # [K,N] NT view (stride0==1)
    assert B_nt.stride()[0] == 1, B_nt.stride()
    w_fp16 = W.to(torch.float16)

    for M in (1, 2048):
        tag = "DECODE" if M == 1 else f"PREFILL M={M}"
        print(f"  --- {tag} ---", flush=True)
        x = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
        ref = (x.to(torch.float32) @ w_fp16.t().to(torch.float32))
        tb = bench(lambda: x @ w_fp16.t())
        print(f"    bf16 matmul                {tb:8.4f} ms  1.00x", flush=True)

        # ---- NEW: int8_gemm_w8a16 (f16 act x s8 weight, fused dequant) ----
        ok = False
        for scd, sc in [("Bsc[N] f16", wscale.to(torch.float16)),
                        ("Bsc[N] f32", wscale.to(torch.float32)),
                        ("Bsc[1,N] f16", wscale.reshape(1, N).to(torch.float16))]:
            try:
                y = call_w8a16(x, B_nt, sc)
                fin = torch.isfinite(y).all().item()
                rel = ((y.to(torch.float32) - ref).norm() / ref.norm()).item()
                if not fin or rel > 0.1:
                    last = f"finite={fin} rel={rel:.2e}"; continue
                t = bench(lambda: call_w8a16(x, B_nt, sc))
                print(f"    int8_gemm_w8a16 ({scd:11s}){t:8.4f} ms  {tb/t:5.2f}x  finite={fin} relerr={rel:.2e}  <== DECODE op", flush=True)
                ok = True; w8a16_sc = sc; break
            except Exception as e:
                last = repr(e)[:160]
        if not ok:
            print(f"    int8_gemm_w8a16 FAILED: {last}", flush=True); w8a16_sc = None

        # ---- NEW: int8_gemm_w8a16 XPUGraph-captured (the production decode path) ----
        if M == 1 and ok:
            try:
                xs = x.clone()
                for _ in range(10): _ = call_w8a16(xs, B_nt, w8a16_sc)
                sync()
                g = torch.xpu.XPUGraph()
                with torch.xpu.graph(g):
                    yout = call_w8a16(xs, B_nt, w8a16_sc)
                sync(); xs.copy_(x); g.replay(); sync()
                relc = ((yout.to(torch.float32) - ref).norm() / ref.norm()).item()
                tc = bench(lambda: g.replay())
                print(f"    int8_gemm_w8a16 GRAPH      {tc:8.4f} ms  {tb/tc:5.2f}x  relerr={relc:.2e}  <== DECODE captured", flush=True)
            except Exception as e:
                print(f"    int8_gemm_w8a16 GRAPH FAILED: {repr(e)[:160]}", flush=True)

        # ---- NEW: int8_gemm_w8a8 (s8 act x s8 weight, fused per-token x per-channel scale) ----
        # external per-token int8 act-quant (or the fused op below)
        amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
        xsc = (amax / 127.0)
        xq = torch.round(x / xsc).clamp_(-127, 127).to(torch.int8)
        ok8 = False
        for scd, (asc, bsc) in [("A[M,1]f16,B[N]f16", (xsc.to(torch.float16), wscale.to(torch.float16))),
                                ("A[M,1]f32,B[N]f32", (xsc.to(torch.float32), wscale.to(torch.float32))),
                                ("A[M,1]f32,B[1,N]f32", (xsc.to(torch.float32), wscale.reshape(1, N).to(torch.float32)))]:
            try:
                y = ops.int8_gemm_w8a8(xq, asc, None, B_nt, bsc, None, None, torch.float16)
                fin = torch.isfinite(y).all().item()
                rel = ((y.to(torch.float32) - ref).norm() / ref.norm()).item()
                if not fin or rel > 0.1:
                    last8 = f"finite={fin} rel={rel:.2e}"; continue
                t = bench(lambda: ops.int8_gemm_w8a8(xq, asc, None, B_nt, bsc, None, None, torch.float16))
                print(f"    int8_gemm_w8a8 ({scd:18s}){t:8.4f} ms  {tb/t:5.2f}x  relerr={rel:.2e}  (op-only, ext act-q)  <== PREFILL op", flush=True)
                ok8 = True; break
            except Exception as e:
                last8 = repr(e)[:160]
        if not ok8:
            print(f"    int8_gemm_w8a8 FAILED: {last8}", flush=True)

        # ---- fused dynamic_per_token_int8_quant (the act-quant launch) ----
        if hasattr(ops, "dynamic_per_token_int8_quant"):
            try:
                tq = bench(lambda: ops.dynamic_per_token_int8_quant(x, True, 8))
                print(f"    dyn_per_token_int8_quant   {tq:8.4f} ms  (fused act-quant launch)", flush=True)
            except Exception as e:
                print(f"    dyn_quant FAILED: {repr(e)[:120]}", flush=True)

        # ---- fp8_gemm_w8a16 bar ----
        if hasattr(ops, "fp8_gemm_w8a16"):
            try:
                fs = (W.abs().amax(1, keepdim=True).clamp_(min=1e-8) / 448.0)
                Wf8 = (W / fs).clamp_(-448, 448).to(torch.float8_e4m3fn)
                Bf8 = Wf8.t(); Bfs = fs.reshape(N).to(torch.float16)
                tf = bench(lambda: ops.fp8_gemm_w8a16(x, Bf8, Bfs, None))
                print(f"    fp8_gemm_w8a16 (the bar)   {tf:8.4f} ms  {tb/tf:5.2f}x", flush=True)
            except Exception as e:
                print(f"    fp8 bar FAILED: {repr(e)[:100]}", flush=True)

print("\nGATE: int8_gemm_w8a16 (decode, captured) ~ fp8 bar (~1.9x); int8_gemm_w8a8 (prefill) realizes 1.7-1.9x XMX.", flush=True)
