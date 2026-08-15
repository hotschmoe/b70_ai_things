#!/usr/bin/env python3
"""Unsloth apply-path isolation (XPU). Compare live kernels vs dequant reference.

FP8:   torch.ops._xpu_C.fp8_gemm_w8a16 vs F.linear(f8 * channel_scale)
NVFP4: torch.ops._xpu_C.nvfp4_gemm_w4a16 vs F.linear(e2m1 * block * 1/wgs)

No vLLM load. Same fused _xpu_C as the one-card Unsloth serve.
"""
from __future__ import annotations

import os
import sys

import torch
from safetensors import safe_open

# Mounted fused _xpu_C.abi3.so lives in vllm_xpu_kernels; import registers the ops.
try:
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    print("imported vllm_xpu_kernels._xpu_C", flush=True)
except Exception as e:
    print("import _xpu_C failed:", type(e).__name__, e, flush=True)

DEV = "xpu"
UNSLOTH = "/models/qwen3.8-27b/nvfp4-unsloth/model.safetensors"
MODELOPT_DIR = "/models/qwen3.8-27b/nvfp4-modelopt"
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def banner(s):
    print("\n===", s, "===", flush=True)


def stats(name, y, ref):
    yf = y.float().reshape(-1)
    rf = ref.float().reshape(-1)
    if yf.numel() != rf.numel():
        print(f"  {name:28s} SHAPE {tuple(y.shape)} vs {tuple(ref.shape)}", flush=True)
        return None
    if not torch.isfinite(yf).all():
        print(f"  {name:28s} NONFINITE nans={int((~torch.isfinite(yf)).sum())}", flush=True)
        return None
    cos = float(torch.nn.functional.cosine_similarity(yf, rf, dim=0))
    rmse = float(torch.sqrt(torch.mean((yf - rf) ** 2)))
    mx = float((yf - rf).abs().max())
    rn = float(rf.norm())
    rel = rmse / (rn / (rf.numel() ** 0.5) + 1e-12)
    print(
        f"  {name:28s} cos={cos:7.4f} rmse={rmse:10.5f} maxabs={mx:10.5f} "
        f"rel={rel:.3e} |y|={float(yf.norm()):.4g} |ref|={rn:.4g}",
        flush=True,
    )
    return cos


def load(path, key, device="cpu"):
    with safe_open(path, framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    return t.to(device)


def dequant_nvfp4(packed, scale_f8, wgs, invert=True):
    # packed [N,K/2] uint8, scale [N,K/16] f8, wgs scalar
    N, kh = packed.shape
    K = kh * 2
    p = packed.to(torch.uint8)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(N, K).long()
    lut = E2M1.to(packed.device)
    w = lut[nib]
    s = scale_f8.to(torch.float32)
    g = float(wgs.reshape(-1)[0].float())
    if invert:
        g = 1.0 / g
    s = s.repeat_interleave(16, dim=1) * g
    return (w * s).to(torch.bfloat16)


def modelopt_shard(key):
    import json
    from pathlib import Path
    idx = json.loads((Path(MODELOPT_DIR) / "model.safetensors.index.json").read_text())
    return str(Path(MODELOPT_DIR) / idx["weight_map"][key])


def fp8_probe():
    banner("FP8 channel: Unsloth layer0.linear_attn.in_proj_qkv")
    prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    w_f8 = load(UNSLOTH, prefix + ".weight")  # [N,K] f8
    sc = load(UNSLOTH, prefix + ".weight_scale")  # [N,1] bf16
    print("weight", tuple(w_f8.shape), w_f8.dtype, "scale", tuple(sc.shape), sc.dtype,
          "scale min/max", float(sc.min()), float(sc.max()), flush=True)
    N, K = w_f8.shape
    w_deq = (w_f8.float() * sc.float()).to(torch.bfloat16)  # [N,K]
    torch.manual_seed(0)
    for M in (1, 8):
        x = torch.randn(M, K, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_deq)
        print(f"-- M={M} x {tuple(x.shape)} ref {tuple(ref.shape)}", flush=True)
        xd = x.to(DEV)
        wd = w_f8.to(DEV)
        sd = sc.to(DEV)
        # Layouts the XPU kernel + CT scheme actually use: weight [K,N], scale [1,N]
        layouts = [
            ("W[K,N] sc[1,N]  (CT+XPU)", wd.t().contiguous(), sd.t().contiguous()),
            ("W[K,N] sc[N]    ", wd.t().contiguous(), sd.reshape(N).contiguous()),
            ("W[K,N] sc[N,1]  ", wd.t().contiguous(), sd.contiguous()),
            ("W[N,K] sc[N,1]  ", wd.contiguous(), sd.contiguous()),
        ]
        if not hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):
            print("  NO fp8_gemm_w8a16", flush=True)
            return
        y_ct = None
        for name, W, S in layouts:
            try:
                y = torch.ops._xpu_C.fp8_gemm_w8a16(xd, W, S, None)
                torch.xpu.synchronize()
                stats(name + " vs ch-ref", y.cpu(), ref)
                if y_ct is None:
                    y_ct = y.cpu()
            except Exception as e:
                print(f"  {name:28s} EXC {type(e).__name__}: {str(e)[:160]}", flush=True)
        if y_ct is not None and M == 1:
            # What is the kernel actually doing?
            scf = sc.float()
            alts = {
                "ref mean-scale": torch.nn.functional.linear(x, (w_f8.float() * scf.mean()).to(torch.bfloat16)),
                "ref max-scale": torch.nn.functional.linear(x, (w_f8.float() * scf.max()).to(torch.bfloat16)),
                "ref min-scale": torch.nn.functional.linear(x, (w_f8.float() * scf.min()).to(torch.bfloat16)),
                "ref f8-raw": torch.nn.functional.linear(x, w_f8.float().to(torch.bfloat16)),
            }
            print("  -- kernel vs alternate refs --", flush=True)
            for n, r in alts.items():
                stats(n, y_ct, r)

    banner("FP8 control: 3.6 ModelOpt q_proj (per-tensor scale)")
    try:
        import json
        from pathlib import Path
        m36 = Path("/models/qwen3.6-27b/nvfp4-modelopt")
        idx = json.loads((m36 / "model.safetensors.index.json").read_text())
        key = "model.language_model.layers.11.self_attn.q_proj.weight"
        shard = str(m36 / idx["weight_map"][key])
        w36 = load(shard, key)
        s36 = load(str(m36 / idx["weight_map"][key.replace(".weight", ".weight_scale")]),
                   key.replace(".weight", ".weight_scale"))
        print("3.6 q_proj", tuple(w36.shape), w36.dtype, "scale", tuple(s36.shape),
              s36.dtype, float(s36.reshape(-1)[0]), flush=True)
        N, K = w36.shape
        w_deq = (w36.float() * float(s36.reshape(-1)[0])).to(torch.bfloat16)
        x = torch.randn(1, K, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_deq)
        y = torch.ops._xpu_C.fp8_gemm_w8a16(
            x.to(DEV), w36.to(DEV).t().contiguous(),
            s36.reshape(1).to(DEV).float(), None
        )
        torch.xpu.synchronize()
        stats("3.6 tensor-scale vs ref", y.cpu(), ref)
    except Exception as e:
        print("  3.6 control skip", type(e).__name__, str(e)[:200], flush=True)


def nvfp4_probe():
    banner("NVFP4: Unsloth layer0.mlp.gate_proj")
    prefix = "model.language_model.layers.0.mlp.gate_proj"
    packed = load(UNSLOTH, prefix + ".weight_packed")
    scale = load(UNSLOTH, prefix + ".weight_scale")
    wgs = load(UNSLOTH, prefix + ".weight_global_scale")
    print("packed", tuple(packed.shape), packed.dtype, "scale", tuple(scale.shape),
          scale.dtype, "wgs", float(wgs.reshape(-1)[0]), flush=True)
    w_deq = dequant_nvfp4(packed, scale, wgs, invert=True)
    N, K = w_deq.shape
    torch.manual_seed(1)
    if not hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"):
        print("  NO nvfp4_gemm_w4a16", flush=True)
        return
    for M in (1, 8):
        x = torch.randn(M, K, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_deq)
        print(f"-- M={M}", flush=True)
        xd = x.to(DEV)
        # fused sitecustomize: B = weight[N,K/2].t() -> [K/2,N], scale_nt = (f8*g).t() [K/16,N]
        g = 1.0 / float(wgs.reshape(-1)[0].float())
        wsc = (scale.float() * g).to(torch.bfloat16)  # [N, K/16]
        B = packed.to(DEV)
        variants = [
            ("inv B.t sc.t gs16", B.t(), wsc.to(DEV).t().contiguous(), 16),
            ("inv B    sc.t gs16", B, wsc.to(DEV).t().contiguous(), 16),
            ("noinv B.t sc.t", B.t(), scale.to(DEV).float().to(torch.bfloat16).t().contiguous(), 16),
        ]
        for name, W, S, gs in variants:
            try:
                y = torch.ops._xpu_C.nvfp4_gemm_w4a16(xd, W, None, S, gs)
                torch.xpu.synchronize()
                stats(name, y.cpu(), ref)
            except Exception as e:
                print(f"  {name:28s} EXC {type(e).__name__}: {str(e)[:160]}", flush=True)

    banner("NVFP4 control: Inferact ModelOpt same layer")
    mp = "model.language_model.layers.0.mlp.gate_proj."
    try:
        mw = load(modelopt_shard(mp + "weight"), mp + "weight")
        ms = load(modelopt_shard(mp + "weight_scale"), mp + "weight_scale")
        ms2 = load(modelopt_shard(mp + "weight_scale_2"), mp + "weight_scale_2")
    except Exception as e:
        print("  modelopt skip", type(e).__name__, e, flush=True)
        return
    print("modelopt", tuple(mw.shape), "s2", float(ms2.reshape(-1)[0]), flush=True)
    w_deq = dequant_nvfp4(mw, ms, ms2, invert=False)
    N, K = w_deq.shape
    x = torch.randn(1, K, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_deq)
    g = float(ms2.reshape(-1)[0].float())
    wsc = (ms.float() * g).to(torch.bfloat16)
    try:
        y = torch.ops._xpu_C.nvfp4_gemm_w4a16(
            x.to(DEV), mw.to(DEV).t(), None, wsc.to(DEV).t().contiguous(), 16
        )
        torch.xpu.synchronize()
        stats("modelopt fused vs dequant", y.cpu(), ref)
    except Exception as e:
        print("  modelopt EXC", type(e).__name__, str(e)[:160], flush=True)


def main():
    print("torch", torch.__version__, "xpu", torch.xpu.is_available(),
          "n", torch.xpu.device_count() if torch.xpu.is_available() else 0, flush=True)
    print("ops fp8_w8a16", hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"),
          "nvfp4_w4a16", hasattr(torch.ops._xpu_C, "nvfp4_gemm_w4a16"), flush=True)
    if not torch.xpu.is_available():
        sys.exit("no xpu")
    fp8_probe()
    nvfp4_probe()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
