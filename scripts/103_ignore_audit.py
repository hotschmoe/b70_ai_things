#!/usr/bin/env python3
# Test codex hypothesis: are GDN linear_attn (+ other ignore-list) layers stored BF16 in the W8A8 ckpt,
# and does the served quant config's ignore/targets correctly exempt them under the language_model nesting?
import json, glob, os
from safetensors import safe_open
import torch

D = "/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft"
cfg = json.load(open(os.path.join(D,"config.json")))
qc = cfg.get("quantization_config", {})
print("=== quantization_config top-level keys ===", list(qc.keys()))
print("ignore:", json.dumps(qc.get("ignore"), indent=0))
cg = qc.get("config_groups", {})
for gname, g in cg.items():
    print(f"group {gname}: targets={g.get('targets')}  weights={ {k:g['weights'].get(k) for k in ('num_bits','strategy','type','symmetric')} if g.get('weights') else None}")

# index all keys -> file
idx = {}
for f in sorted(glob.glob(os.path.join(D,"*.safetensors"))):
    with safe_open(f, framework="pt", device="cpu") as sf:
        for k in sf.keys(): idx[k]=f
def dtype_of(k):
    f=idx.get(k)
    if not f: return None
    with safe_open(f, framework="pt", device="cpu") as sf:
        return str(sf.get_slice(k).get_dtype())

# find first GDN (linear_attn) layer and first full-attn (self_attn) layer
import re
la_layers = sorted({int(re.search(r"layers\.(\d+)\.",k).group(1)) for k in idx if "linear_attn" in k})
sa_layers = sorted({int(re.search(r"layers\.(\d+)\.",k).group(1)) for k in idx if "self_attn.q_proj" in k})
print("\nGDN(linear_attn) layers:", la_layers[:6], "...count", len(la_layers))
print("full-attn(self_attn) layers:", sa_layers[:6], "...count", len(sa_layers))

print("\n=== sample linear_attn tensors (dtype + has weight_scale?) ===")
L = la_layers[0]
for k in sorted(idx):
    if f".layers.{L}.linear_attn." in k:
        print(f"  {k}  dtype={dtype_of(k)}")

print("\n=== sample full-attn self_attn.q_proj (dtype + scale) ===")
S = sa_layers[0]
for k in sorted(idx):
    if f".layers.{S}.self_attn.q_proj" in k:
        print(f"  {k}  dtype={dtype_of(k)}")

# the crux: does the ignore list use names that match 'model.language_model.layers.N.linear_attn...'?
print("\n=== ignore-match test ===")
ig = qc.get("ignore", []) or []
sample_la = f"model.language_model.layers.{L}.linear_attn.in_proj_qkvz"
sample_la2 = f"language_model.layers.{L}.linear_attn.in_proj_qkvz"
sample_la3 = f"model.layers.{L}.linear_attn.in_proj_qkvz"
def matches(name):
    out=[]
    for pat in ig:
        if pat.startswith("re:"):
            if re.search(pat[3:], name): out.append(pat)
        else:
            if pat==name or name.endswith(pat): out.append(pat)
    return out
for nm in (sample_la, sample_la2, sample_la3):
    print(f"  '{nm}' -> ignore matches: {matches(nm)}")
print("DONE")
