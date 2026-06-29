#!/usr/bin/env python3
# w8a16_decode_probe.py -- is there a TORCH-NATIVE int8 W8A16 decode GEMV on XPU?
#
# The W8A8 decode gap (campaign doc sec 3): the 3-kernel int8 chain (act-quant -> _int_mm ->
# dequant) loses to the single fused fp8_gemm_w8a16 (1.95x) even when XPUGraph-captured. The
# clean fix is a 1-launch int8 W8A16 op (int8 weight, fp16 act, per-channel scale, dequant in
# kernel). Before building one in oneDNN (~50min AOT), check if torch ALREADY exposes it on XPU:
#   - torch._weight_int8pack_mm(A_fp16[M,K], B_int8[N,K], scales[N]) -> [M,N]   (torchao/gpt-fast)
#   - torch.ops.aten._weight_int8pack_mm
#   - any ipex / onednn weights-decompress path
# If yes -> instant decode op, no build. If no -> proceed with the custom oneDNN op.
import os, sys, time, ctypes, traceback
import torch

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/w4a8_kernel/_xpu_C.abi3.so")
print("torch", torch.__version__, "xpu", torch.xpu.is_available(), flush=True)
try:
    ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL); HAVE_SO = True
    print("CDLL OK", flush=True)
except OSError as e:
    HAVE_SO = False; print("CDLL fail:", str(e)[:120], flush=True)
HAVE_FP8 = HAVE_SO and hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")

print("\n== op availability ==", flush=True)
print("  torch._weight_int8pack_mm:", hasattr(torch, "_weight_int8pack_mm"), flush=True)
print("  aten._weight_int8pack_mm :", hasattr(torch.ops.aten, "_weight_int8pack_mm"), flush=True)
print("  torch._int_mm            :", hasattr(torch, "_int_mm"), flush=True)
try:
    import intel_extension_for_pytorch as ipex
    print("  ipex:", ipex.__version__, flush=True)
except Exception as e:
    print("  ipex: not importable", str(e)[:60], flush=True)


def sync(): torch.xpu.synchronize()
def bench(fn, warm=30, iters=80):
    for _ in range(warm): fn()
    sync(); s = time.time()
    for _ in range(iters): fn()
    sync(); return (time.time() - s) / iters * 1000.0


SHAPES = [("gate_up", 34816, 5120), ("down_proj", 5120, 17408), ("qkv", 14336, 5120)]

for (name, N, K) in SHAPES:
    print(f"\n===== {name} [N={N} K={K}] DECODE M=1 =====", flush=True)
    W = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02)
    amax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)        # [N,1]
    wscale = (amax / 127.0)                                          # [N,1]
    Wq = torch.round(W / wscale).clamp_(-127, 127).to(torch.int8)    # [N,K] int8
    scales_N = wscale.reshape(N)                                     # [N]
    x = torch.randn(1, K, device=DEV, dtype=torch.float16) * 0.05
    w_fp16 = W.to(torch.float16)
    ref = (x.to(torch.float32) @ w_fp16.t().to(torch.float32))

    tb = bench(lambda: x @ w_fp16.t())
    print(f"  bf16 matmul           {tb:.4f} ms  1.00x (baseline)", flush=True)

    # torch._weight_int8pack_mm: B=[N,K] int8 (row-major), scales=[N] (A.dtype)
    for opname, op in [("_weight_int8pack_mm", getattr(torch, "_weight_int8pack_mm", None)),
                       ("aten._weight_int8pack_mm", getattr(torch.ops.aten, "_weight_int8pack_mm", None))]:
        if op is None:
            continue
        ok = False
        for sc_desc, sc in [("scales[N] fp16", scales_N.to(torch.float16)),
                            ("scales[N] fp32", scales_N.to(torch.float32)),
                            ("scales[N] bf16", scales_N.to(torch.bfloat16))]:
            for bdesc, B in [("B[N,K]", Wq), ("B[N,K].contig", Wq.contiguous())]:
                try:
                    y = op(x, B, sc)
                    fin = torch.isfinite(y).all().item()
                    rel = ((y.to(torch.float32) - ref).norm() / ref.norm()).item()
                    if not fin or rel > 0.2:
                        continue
                    t = bench(lambda: op(x, B, sc))
                    print(f"  {opname} ({sc_desc},{bdesc})  {t:.4f} ms  {tb/t:.2f}x  finite={fin} relerr={rel:.2e}  <== WIN", flush=True)
                    ok = True; break
                except Exception as e:
                    last = repr(e)[:140]
            if ok: break
        if not ok:
            print(f"  {opname}: FAILED all layouts -- {last}", flush=True)

    # fp8 w8a16 reference (the bar)
    if HAVE_FP8:
        try:
            wamax = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
            fs = (wamax / 448.0); Wf8 = (W / fs).clamp_(-448, 448).to(torch.float8_e4m3fn)
            B8 = Wf8.t(); Bsc = fs.reshape(N).to(torch.float16)
            y = torch.ops._xpu_C.fp8_gemm_w8a16(x, B8, Bsc, None)
            t = bench(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(x, B8, Bsc, None))
            print(f"  fp8_gemm_w8a16        {t:.4f} ms  {tb/t:.2f}x  (the bar to match w/ int8)", flush=True)
        except Exception as e:
            print("  fp8 w8a16 fail:", repr(e)[:100], flush=True)

print("\nGATE: if _weight_int8pack_mm runs FINITE + FAST on XPU -> instant int8 W8A16 decode op (no build).", flush=True)
