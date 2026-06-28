#!/usr/bin/env python3
# w4a8_triton_aq_probe.py -- validate + bench the Triton per-token int8 act-quant vs the eager chain,
# then verify int4_gemm_w4a8 output is unchanged. Run INSIDE sglang-xpu:mtp on card 0 (oneAPI sourced,
# LD_LIBRARY_PATH prepended, B70_XPU_C_SO set). Uses a real sqgptq prepacked layer (K=17408,N=5120).
import os, sys, time, ctypes, torch
sys.path.insert(0, "/work/patches")
import w4a8_actquant_triton as T

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/w4a8_kernel/_xpu_C.abi3.so")
ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
assert hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"), "op not registered"
print("triton available:", T.available())
if not T.available():
    print("TRITON IMPORT FAILED:", getattr(T, "_TRITON_ERR", "?")); sys.exit(1)

def eager_q(x):
    amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
    xs = (amax / 127.0).to(torch.float16)
    xq = (x / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
    xz = torch.zeros_like(amax, dtype=torch.int32).contiguous()
    return xq, xs.contiguous(), xz

import safetensors.torch as stt
CKPT = "/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
PFX = "model.language_model.layers.20.mlp.down_proj"
t = stt.load_file(CKPT)
wq = t[f"{PFX}.weight"].to(DEV); ws = t[f"{PFX}.weight_scale"].to(DEV)
N, K8 = wq.shape; K = K8 * 8; G = 128
qweight = wq.t(); wscale = ws.t().contiguous(); wzp = torch.tensor([8], dtype=torch.int8, device=DEV)
print(f"layer down_proj N={N} K={K}")

def sync(): torch.xpu.synchronize()
def bench(fn, iters=60, warm=25):
    for _ in range(warm): fn()
    sync(); s = time.time()
    for _ in range(iters): fn()
    sync(); return (time.time() - s) / iters * 1000.0

print("\n==== numerics (eager vs triton act-quant) ====")
for M in (1, 512, 2048):
    x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1).contiguous()
    eq, es, ez = eager_q(x)
    tq, ts, tz = T.per_token_int8(x)
    qdiff = (eq.to(torch.int16) - tq.to(torch.int16)).abs()
    sdiff = (es.float() - ts.float()).abs().max().item()
    nmis = (qdiff > 0).sum().item()
    print(f"M={M:>4} q max|diff|={qdiff.max().item()} mismatches={nmis}/{M*K} ({100*nmis/(M*K):.3f}%) scale max|diff|={sdiff:.3e}")
    # op output equivalence
    ye = torch.ops._xpu_C.int4_gemm_w4a8(eq, es, ez, qweight, wscale, wzp, G, None, None)
    yt = torch.ops._xpu_C.int4_gemm_w4a8(tq, ts, tz, qweight, wscale, wzp, G, None, None)
    rel = ((ye.float() - yt.float()).norm() / ye.float().norm().clamp_min(1e-9)).item()
    print(f"      int4_gemm_w4a8 out relerr(eager vs triton) = {rel:.3e}  finite={torch.isfinite(yt).all().item()}")

print("\n==== bench (warm, ms/call) ====")
for M in (512, 2048):
    x = (torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.1).contiguous()
    t_eaq = bench(lambda: eager_q(x))
    t_taq = bench(lambda: T.per_token_int8(x))
    def full_e():
        q, s, z = eager_q(x); return torch.ops._xpu_C.int4_gemm_w4a8(q, s, z, qweight, wscale, wzp, G, None, None)
    def full_t():
        q, s, z = T.per_token_int8(x); return torch.ops._xpu_C.int4_gemm_w4a8(q, s, z, qweight, wscale, wzp, G, None, None)
    t_fe = bench(full_e); t_ft = bench(full_t)
    print(f"M={M:>4}  act-quant: eager={t_eaq:.4f}ms triton={t_taq:.4f}ms ({t_eaq/t_taq:.2f}x faster)  |  "
          f"full w4a8: eager={t_fe:.4f}ms triton={t_ft:.4f}ms ({t_fe/t_ft:.2f}x faster)")
print("\nGATE: q mismatches <1% + op relerr <1e-2 + triton act-quant faster -> wire into shim.")
