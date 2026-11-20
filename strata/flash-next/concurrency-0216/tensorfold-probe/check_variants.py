#!/usr/bin/env python3
"""Structure + numeric sanity of rotate_w4.py outputs. usage: check_variants.py BF16_DIR GPTQ_TEMPLATE VARIANT_DIR rot|norot"""
import json, os, random, sys
import torch
from safetensors import safe_open
bf, gq, var, mode = sys.argv[1:5]
H = torch.ones(1, 1)
while H.shape[0] < 128: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = H / 128 ** 0.5
gi = json.load(open(f"{gq}/model.safetensors.index.json"))["weight_map"]
vi = json.load(open(f"{var}/model.safetensors.index.json"))["weight_map"]
bi = json.load(open(f"{bf}/model.safetensors.index.json"))["weight_map"]
print("index equal to GPTQ template:", gi == vi, "| tensors", len(vi), "| marker hadamard128:", os.path.exists(f"{var}/hadamard128"), "(expected %s)" % (mode == "rot"))
def get(d, idx, k):
    with safe_open(f"{d}/{idx[k]}", "pt") as f: return f.get_tensor(k)
def deq(qw, sc):
    k8, n = qw.shape; k = k8 * 8
    q = torch.stack([(qw >> (4 * j)) & 15 for j in range(8)], 1).view(k, n)   # [K, N]
    return ((q.float() - 8) * sc.float().repeat_interleave(128, 0)).t()      # [N, K]
random.seed(0)
names = sorted(k[:-8] for k in vi if k.endswith(".qweight")); print("quantized projections:", len(names))
for base in random.sample(names, 6):
    w = get(bf, bi, base + ".weight").float()
    n, k = w.shape
    ref = (w.view(n, k // 128, 128) @ H).view(n, k) if mode == "rot" else w
    ev = (deq(get(var, vi, base + ".qweight"), get(var, vi, base + ".scales")) - ref).norm() / ref.norm()
    eg = (deq(get(gq, gi, base + ".qweight"), get(gq, gi, base + ".scales")) - w).norm() / w.norm()
    print(f"{base}: rel err vs {'rotated ' if mode=='rot' else ''}source: RTN variant {ev:.4f} | SergiioB GPTQ vs plain W {eg:.4f}")
# non-projection tensors that must equal the template bit for bit (embed, head, norms, mtp), except rotated in_proj_a/b
bad = 0; chk = [k for k in vi if not k.endswith((".qweight", ".scales", ".qzeros", ".g_idx"))]
for k in random.sample(chk, min(60, len(chk))):
    if mode == "rot" and k.endswith(("in_proj_a.weight", "in_proj_b.weight")): continue
    if not torch.equal(get(var, vi, k), get(gq, gi, k)): bad += 1; print("DIFFERS from template:", k)
print("pass-through tensors sampled:", min(60, len(chk)), "differing:", bad)
