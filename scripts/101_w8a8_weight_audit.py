#!/usr/bin/env python3
# Audit the 27B W8A8-sqgptq weights: are they CORRUPT (bad quant) or FINE-but-skipped-at-load?
# Dequantize a few int8 linears and compare per-output-channel cosine sim vs the bf16 base.
# CPU only, safetensors lazy-load (cheap). Run inside vllm-xpu-env:int8g.
import sys, glob, os
import torch
from safetensors import safe_open

W8 = "/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft"
BASE = "/models/Qwen_Qwen3.6-27B"

def open_files(d):
    fs = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    return fs

def build_index(files):
    idx = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                idx[k] = f
    return idx

print("=== W8A8 files ==="); w8files = open_files(W8); print("\n".join(w8files))
print("=== BASE files (count) ==="); bfiles = open_files(BASE); print(len(bfiles))
w8idx = build_index(w8files)
bidx = build_index(bfiles)

# show layer-0 key naming in each
print("\n=== W8A8 layer.0 keys (sample) ===")
for k in sorted(w8idx):
    if ".layers.0." in k and ("down_proj" in k or "q_proj" in k):
        print("  ", k)
print("=== BASE layer.0 keys (sample) ===")
for k in sorted(bidx):
    if ".layers.0." in k and ("down_proj" in k or "q_proj" in k):
        print("  ", k)

def get(idx, key):
    f = idx.get(key)
    if f is None: return None
    with safe_open(f, framework="pt", device="cpu") as sf:
        return sf.get_tensor(key)

def base_key_for(w8key):
    # try a few naming variants in the base
    cand = [w8key,
            w8key.replace("model.layers", "model.language_model.layers"),
            w8key.replace("model.language_model.layers", "model.layers")]
    for c in cand:
        if c in bidx: return c
    return None

print("\n=== dequant vs base (per-output-channel cosine) ===")
layers_to_check = [0, 1, 30, 63]
for L in layers_to_check:
    for proj in ["mlp.down_proj", "self_attn.q_proj", "mlp.gate_proj"]:
        wk = f"model.layers.{L}.{proj}.weight"
        sk = f"model.layers.{L}.{proj}.weight_scale"
        w = get(w8idx, wk); s = get(w8idx, sk)
        if w is None:
            # maybe language_model prefix
            wk2 = wk.replace("model.layers","model.language_model.layers")
            sk2 = sk.replace("model.layers","model.language_model.layers")
            w = get(w8idx, wk2); s = get(w8idx, sk2)
            if w is not None: wk = wk2
        if w is None:
            print(f"  L{L} {proj}: W8 weight MISSING"); continue
        bk = base_key_for(wk)
        b = get(bidx, bk) if bk else None
        # stats
        wq = w.to(torch.int32)
        sat = ((wq==127)|(wq==-128)).float().mean().item()
        nan_s = torch.isnan(s.float()).any().item() if s is not None else "noscale"
        if s is not None:
            deq = w.float() * s.float().reshape(-1,1) if s.numel()==w.shape[0] else w.float()*float(s.float().mean())
        else:
            deq = w.float()
        msg = f"  L{L} {proj}: int8 sat%={sat*100:.2f} scale_nan={nan_s} scale_shape={tuple(s.shape) if s is not None else None}"
        if b is not None and b.shape==deq.shape:
            bf = b.float()
            # per-output-channel cosine
            num = (deq*bf).sum(dim=1)
            den = deq.norm(dim=1)*bf.norm(dim=1)+1e-9
            cos = (num/den)
            msg += f" | base={bk.split('.')[-1] if bk else None} cos_mean={cos.mean().item():.4f} cos_min={cos.min().item():.4f}"
        elif b is not None:
            msg += f" | base shape {tuple(b.shape)} != deq {tuple(deq.shape)}"
        else:
            msg += f" | base key not found ({bk})"
        print(msg)
print("\nDONE")
