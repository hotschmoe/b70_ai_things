#!/usr/bin/env python3
"""Verify a merged checkpoint against its base and adapter (CPU, streaming).
usage: verify_merge.py BASE_DIR ADAPTER_DIR MERGED_DIR [N_BITEXACT_SAMPLE]"""
import json, random, re, sys
import torch
from safetensors import safe_open
base, adir, mer = sys.argv[1:4]
nsamp = int(sys.argv[4]) if len(sys.argv) > 4 else 40
bi = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
mi = json.load(open(f"{mer}/model.safetensors.index.json"))["weight_map"]
assert bi == mi, "index weight_map differs"
mods = set()
with safe_open(f"{adir}/adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        mods.add(re.fullmatch(r"base_model\.model\.(.+)\.lora_[AB]\.weight", k).group(1) + ".weight")
print("adapter modules", len(mods), "all in base:", all(m in bi for m in mods))
cfg = json.load(open(f"{adir}/adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
n = shape_bad = dtype_bad = 0
untouched = []
for shard in sorted(set(bi.values())):
    with safe_open(f"{base}/{shard}", "pt") as fb, safe_open(f"{mer}/{shard}", "pt") as fm:
        assert set(fb.keys()) == set(fm.keys()), shard
        for k in fb.keys():
            n += 1
            sb, sm = fb.get_slice(k), fm.get_slice(k)
            if sb.get_shape() != sm.get_shape(): shape_bad += 1
            if sb.get_dtype() != sm.get_dtype(): dtype_bad += 1
            if k not in mods: untouched.append((shard, k))
print("tensors", n, "shape mismatches", shape_bad, "dtype mismatches", dtype_bad)
random.seed(0); sample = random.sample(untouched, min(nsamp, len(untouched)))
bad = 0
for shard, k in sample:
    with safe_open(f"{base}/{shard}", "pt") as fb, safe_open(f"{mer}/{shard}", "pt") as fm:
        if not torch.equal(fb.get_tensor(k), fm.get_tensor(k)): bad += 1; print("DIFFERS (should be untouched):", k)
vis = [k for _, k in untouched if ".visual." in k or k.startswith("visual.") or "model.visual" in k]
print("untouched tensors", len(untouched), "bit-exact sample", len(sample), "differing", bad, "| vision tensors in checkpoint", len(vis))
# independent recomputation of the delta for a spread of merged tensors
ad = safe_open(f"{adir}/adapter_model.safetensors", "pt")
pick = sorted(mods); random.seed(1); pick = random.sample(pick, 12)
worst = 0
for k in pick:
    pre = "base_model.model." + k[:-7]
    A, B = ad.get_tensor(pre + ".lora_A.weight").float(), ad.get_tensor(pre + ".lora_B.weight").float()
    d = (B @ A) * scale
    shard = bi[k]
    with safe_open(f"{base}/{shard}", "pt") as fb, safe_open(f"{mer}/{shard}", "pt") as fm:
        w, m = fb.get_tensor(k).float(), fm.get_tensor(k).float()
    eff = m - w
    cos = torch.nn.functional.cosine_similarity(eff.flatten(), d.flatten(), dim=0).item()
    ratio = eff.norm().item() / d.norm().item()
    worst = max(worst, abs(1 - ratio))
    print(f"{k}: ||delta||={d.norm():.4f} ||merged-base||={eff.norm():.4f} ratio={ratio:.4f} cos={cos:.5f} rel|W|={d.norm()/w.norm():.5f}")
print("worst |1-ratio| over sample:", round(worst, 5))
st = json.load(open(f"{mer}/merge_stats.json"))
print("merge_stats modules", st["n_modules"], "max err_over_delta", max(v["err_over_delta"] for v in st["stats"].values()),
      "min changed_frac", min(v["changed_frac"] for v in st["stats"].values()))
print("VERIFY OK" if (shape_bad == 0 and dtype_bad == 0 and bad == 0 and len(mods) == st["n_modules"] == 512 // 2) else "VERIFY FAILED")
