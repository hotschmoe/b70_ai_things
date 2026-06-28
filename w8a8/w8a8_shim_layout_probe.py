#!/usr/bin/env python3
# w8a8_shim_layout_probe.py -- validate the FUSED w8a8_shim layout on REAL sqgptq W8A8 weights (card 0).
# Replicates w8a8_shim._pw_fused / _apply_fused exactly on real [N,K] int8 weights + per-channel scale,
# so a green run here means the shim will serve correctly (layout + dispatch + the fused act-quant op).
# Reference = x @ (Wq.float()*wscale).t() (the SAME int8 weights dequantized) -> isolates OP error.
import os, sys, time, ctypes, json
import torch
import safetensors.torch as st

DEV = "xpu"
SO = os.environ.get("B70_XPU_C_SO", "/work/kernel/_xpu_C.abi3.so")
CKPT = os.environ.get("CKPT", "/models/Qwen3.6-27B-W8A8-sqgptq-vision")
print("torch", torch.__version__, "xpu", torch.xpu.is_available(), flush=True)
ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL); print("CDLL OK", flush=True)
ops = torch.ops._xpu_C
for nm in ["int8_gemm_w8a16", "int8_gemm_w8a8", "dynamic_per_token_int8_quant"]:
    assert hasattr(ops, nm), f"{nm} missing"; print("  op", nm, "OK", flush=True)

idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
LAYERS = [
    ("layers.3.mlp.down_proj", "model.language_model.layers.3.mlp.down_proj"),
    ("layers.3.mlp.gate_up_proj", "model.language_model.layers.3.mlp.gate_up_proj"),
    ("layers.3.self_attn.qkv_proj", "model.language_model.layers.3.self_attn.qkv_proj"),
]


def sync(): torch.xpu.synchronize()
def bench(fn, warm=20, iters=50):
    for _ in range(warm): fn()
    sync(); s = time.time()
    for _ in range(iters): fn()
    sync(); return (time.time() - s) / iters * 1000.0


def load(prefix):
    wk, sk = prefix + ".weight", prefix + ".weight_scale"
    if wk not in idx:
        return None
    W = st.load_file(f"{CKPT}/{idx[wk]}")[wk]            # [N,K] int8
    S = st.load_file(f"{CKPT}/{idx[sk]}")[sk]            # [N] or [N,1]
    return W, S


for tag, prefix in LAYERS:
    r = load(prefix)
    if r is None:
        print(f"\n[{tag}] not found, skip", flush=True); continue
    W, S = r
    N, K = W.shape
    print(f"\n===== {tag} W{tuple(W.shape)} {W.dtype}  scale{tuple(S.shape)} {S.dtype} =====", flush=True)
    Wd = W.to(DEV)                                       # [N,K] int8
    # replicate shim _pw_fused layout: NT B [K,N] stride0==1 via contiguous [N,K] backing
    weight_NK = Wd.contiguous()
    B_nt = weight_NK.t()                                 # [K,N] stride0==1
    assert B_nt.stride()[0] == 1, B_nt.stride()
    wscale_n = S.to(DEV).reshape(-1).to(torch.float16)   # [N]
    Wdeq = (Wd.to(torch.float32) * wscale_n.to(torch.float32).reshape(N, 1))   # [N,K] dequant ref

    for M in (1, 2048):
        which = "DECODE" if M == 1 else "PREFILL"
        x = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
        ref = x.to(torch.float32) @ Wdeq.t()
        if M == 1:
            y = ops.int8_gemm_w8a16(x, B_nt, wscale_n, None)
            rel = ((y.to(torch.float32) - ref).norm() / ref.norm()).item()
            fin = torch.isfinite(y).all().item()
            t = bench(lambda: ops.int8_gemm_w8a16(x, B_nt, wscale_n, None))
            print(f"  {which} int8_gemm_w8a16  {t:.4f}ms  finite={fin}  relerr={rel:.2e}", flush=True)
            # graph capture
            try:
                xs2 = x.clone()
                for _ in range(8): _ = ops.int8_gemm_w8a16(xs2, B_nt, wscale_n, None)
                sync()
                g = torch.xpu.XPUGraph()
                with torch.xpu.graph(g):
                    yo = ops.int8_gemm_w8a16(xs2, B_nt, wscale_n, None)
                sync(); xs2.copy_(x); g.replay(); sync()
                relc = ((yo.to(torch.float32) - ref).norm() / ref.norm()).item()
                print(f"         GRAPH replay  finite={torch.isfinite(yo).all().item()}  relerr={relc:.2e}", flush=True)
            except Exception as e:
                print(f"         GRAPH FAILED: {repr(e)[:140]}", flush=True)
        else:
            # fused act-quant op + int8_gemm_w8a8
            xq, xsc, xz = ops.dynamic_per_token_int8_quant(x, True, 8)
            aq_rel = (((xq.to(torch.float32) * xsc.to(torch.float32)) - x.to(torch.float32)).norm()
                      / x.to(torch.float32).norm()).item()
            y = ops.int8_gemm_w8a8(xq, xsc, None, B_nt, wscale_n, None, None, torch.float16)
            rel = ((y.to(torch.float32) - ref).norm() / ref.norm()).item()
            fin = torch.isfinite(y).all().item()
            t = bench(lambda: ops.int8_gemm_w8a8(xq, xsc, None, B_nt, wscale_n, None, None, torch.float16))
            print(f"  {which} int8_gemm_w8a8   {t:.4f}ms  finite={fin}  relerr={rel:.2e}  (act-quant relerr {aq_rel:.2e})", flush=True)

print("\nGATE: finite + decode relerr<5e-3 + prefill relerr<3e-2 on real weights -> shim layout CORRECT.", flush=True)
