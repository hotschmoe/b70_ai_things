#!/usr/bin/env python3
"""Unsloth channel-FP8 -> INT8-XMX microbench (card 1).

Xe2 has no native FP8. fp8_gemm_w8a16 is emulated + per-tensor-only.
Repack f8 * per-channel scale -> s8 + per-out-channel scale, then
int8_gemm_w8a16 (oneDNN INT8 XMX). Compare vs tiled F.linear reference
and vs the broken per-tensor fp8_gemm.
"""
from __future__ import annotations

import os
import time

import torch
from safetensors import safe_open

try:
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    print("imported vllm_xpu_kernels._xpu_C", flush=True)
except Exception as e:
    print("import _xpu_C failed", type(e).__name__, e, flush=True)

DEV = "xpu"
UNSLOTH = "/models/qwen3.8-27b/nvfp4-unsloth/model.safetensors"
PREFIX = "model.language_model.layers.0.linear_attn.in_proj_qkv"


def load(path, key):
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def stats(name, y, ref):
    yf, rf = y.float().reshape(-1), ref.float().reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(yf, rf, dim=0))
    rmse = float(torch.sqrt(torch.mean((yf - rf) ** 2)))
    mx = float((yf - rf).abs().max())
    print(
        f"  {name:28s} cos={cos:7.4f} rmse={rmse:10.5f} maxabs={mx:10.5f} "
        f"|y|={float(yf.norm()):.4g} |ref|={float(rf.norm()):.4g}",
        flush=True,
    )
    return cos


def bench(fn, warm=10, iters=30):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print(
        "torch", torch.__version__,
        "xpu", torch.xpu.is_available(),
        "int8_w8a16", hasattr(torch.ops._xpu_C, "int8_gemm_w8a16"),
        "fp8_w8a16", hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"),
        flush=True,
    )
    if not torch.xpu.is_available():
        raise SystemExit("no xpu")
    w_f8 = load(UNSLOTH, PREFIX + ".weight")  # [N,K]
    sc = load(UNSLOTH, PREFIX + ".weight_scale")  # [N,1]
    n, k = w_f8.shape
    print("layer", PREFIX, "N,K", n, k, "f8", w_f8.dtype, "sc", tuple(sc.shape), flush=True)
    w_deq = (w_f8.float() * sc.float()).to(torch.bfloat16)  # [N,K] true dequant

    # INT8-XMX repack: per-output-channel s8 of (f8 * scale)
    amax = w_deq.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [N,1]
    iscale = (amax / 127.0)  # [N,1]
    w_s8 = torch.round(w_deq.float() / iscale).clamp(-127, 127).to(torch.int8)  # [N,K]
    wt = w_s8.t().contiguous()  # [K,N]
    sc_row = iscale.reshape(1, n).to(torch.bfloat16).contiguous()
    print(
        "repack s8 unique", int(w_s8.unique().numel()),
        "iscale min/max", float(iscale.min()), float(iscale.max()),
        flush=True,
    )

    torch.manual_seed(0)
    have_i8 = hasattr(torch.ops._xpu_C, "int8_gemm_w8a16")
    have_f8 = hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16")

    # Scale-layout sweep (M=1): [1,N] was treated as K-block and matched the
    # broken fp8 mean-scale. Per-channel wants 1D [N] per 03_xmx_microbench.
    if have_i8:
        print("\n=== int8 scale-layout sweep M=1 ===", flush=True)
        x = torch.randn(1, k, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_deq)
        xd = x.to(DEV)
        W = wt.to(DEV)
        layouts = [
            ("[N] 1d", iscale.reshape(n).to(torch.bfloat16).contiguous()),
            ("[N,1]", iscale.to(torch.bfloat16).contiguous()),
            ("[1,N]", iscale.reshape(1, n).to(torch.bfloat16).contiguous()),
            ("[1,1] mean", iscale.mean().reshape(1).to(torch.bfloat16).contiguous()),
        ]
        for name, S in layouts:
            try:
                y = torch.ops._xpu_C.int8_gemm_w8a16(xd, W, S.to(DEV), None)
                torch.xpu.synchronize()
                stats(name, y.cpu(), ref)
            except Exception as e:
                print(f"  {name:28s} EXC {type(e).__name__}: {str(e)[:160]}", flush=True)

    if os.environ.get("SWEEP_ONLY", "0") == "1":
        print("\nSWEEP_ONLY=1, skip timing loop", flush=True)
        print("\nDONE", flush=True)
        return

    for m in (1, 8, 64, 256):
        x = torch.randn(m, k, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_deq)
        print(f"\n-- M={m} --", flush=True)
        xd = x.to(DEV)

        # tiled F.linear (current correctness path)
        def tiled():
            tile = 4096
            outs = []
            xf = xd
            wf = w_f8.to(DEV)
            sf = sc.to(DEV).reshape(-1)
            for n0 in range(0, n, tile):
                n1 = min(n0 + tile, n)
                wt_t = (wf[n0:n1].float() * sf[n0:n1, None].float()).to(torch.bfloat16)
                outs.append(torch.nn.functional.linear(xf, wt_t))
            return torch.cat(outs, dim=-1)

        y_t = tiled()
        torch.xpu.synchronize()
        stats("tiled F.linear", y_t.cpu(), ref)
        t_tile = bench(lambda: tiled())

        t_i8 = None
        if have_i8:
            W = wt.to(DEV)
            S = sc_row.to(DEV)
            y_i8 = torch.ops._xpu_C.int8_gemm_w8a16(xd, W, S, None)
            torch.xpu.synchronize()
            stats("int8_gemm_w8a16 XMX", y_i8.cpu(), ref)
            t_i8 = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(xd, W, S, None))

        t_f8 = None
        if have_f8:
            Wf = w_f8.to(DEV).t().contiguous()
            Sf = sc.to(DEV).t().contiguous()
            y_f8 = torch.ops._xpu_C.fp8_gemm_w8a16(xd, Wf, Sf, None)
            torch.xpu.synchronize()
            stats("fp8_gemm (broken ch)", y_f8.cpu(), ref)
            t_f8 = bench(lambda: torch.ops._xpu_C.fp8_gemm_w8a16(xd, Wf, Sf, None))

        msg = f"  TIME tiled={t_tile:.3f} ms"
        if t_i8 is not None:
            msg += f"  int8xmx={t_i8:.3f} ms ({t_tile / t_i8:.2f}x vs tiled)"
        if t_f8 is not None:
            msg += f"  fp8emul={t_f8:.3f} ms"
        print(msg, flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
