#!/usr/bin/env python3
"""Streaming per-tensor LoRA merge into a bf16 safetensors checkpoint (CPU only).

W' = bf16( float32(W) + (lora_alpha / r) * (B @ A) )   per target module, shard by shard.
Stops (exit 3) on DoRA, rank/alpha patterns, rslora (handled), lora_bias, modules_to_save, or any unmatched module.
usage: merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR
"""
import json, os, re, shutil, sys, time, hashlib
import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(4)
base, adir, out = sys.argv[1:4]
cfg = json.load(open(f"{adir}/adapter_config.json"))

def stop(msg):
    print("STOP:", msg, file=sys.stderr); sys.exit(3)

if cfg.get("peft_type") != "LORA": stop(f"peft_type {cfg.get('peft_type')}")
for k in ("use_dora", "use_qalora", "use_bdlora", "lora_bias"):
    if cfg.get(k): stop(f"{k} is set")
for k in ("rank_pattern", "alpha_pattern"):
    if cfg.get(k): stop(f"{k} not empty")
for k in ("modules_to_save", "trainable_token_indices", "layer_replication", "target_parameters", "exclude_modules"):
    if cfg.get(k): stop(f"{k} = {cfg.get(k)}")
if cfg.get("bias") != "none": stop(f"bias {cfg.get('bias')}")
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
print(f"r={r} alpha={alpha} rslora={cfg.get('use_rslora')} scale={scale}", flush=True)

# adapter tensors: base_model.model.<module>.lora_A.weight -> <module>.weight
ad = {}
with safe_open(f"{adir}/adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", k)
        if not m: stop(f"unexpected adapter key {k}")
        ad.setdefault(m.group(1) + ".weight", {})[m.group(2)] = f.get_tensor(k)
        if f.get_slice(k).get_dtype() not in ("F32", "F16", "BF16"): stop("adapter dtype")
print("adapter modules:", len(ad), flush=True)

idx = json.load(open(f"{base}/model.safetensors.index.json"))
wm = idx["weight_map"]
miss = [n for n in ad if n not in wm]
if miss: stop(f"{len(miss)} adapter modules not in base, e.g. {miss[:3]}")

os.makedirs(out, exist_ok=True)
stats, used = {}, set()
t0 = time.time()
for shard in sorted(set(wm.values())):
    tens = {}
    with safe_open(f"{base}/{shard}", "pt") as f:
        meta = f.metadata()
        for n in f.keys():
            w = f.get_tensor(n)
            if n in ad:
                A, B = ad[n]["A"].float(), ad[n]["B"].float()
                assert w.dtype == torch.bfloat16 and w.shape == (B.shape[0], A.shape[1]) and A.shape[0] == B.shape[1] == r, (n, w.shape, A.shape, B.shape)
                delta = (B @ A) * scale
                merged = (w.float() + delta).to(torch.bfloat16)
                eff = (merged.float() - w.float())
                stats[n] = dict(delta_norm=delta.norm().item(), eff_norm=eff.norm().item(),
                                err_over_delta=(eff - delta).norm().item() / max(delta.norm().item(), 1e-30),
                                w_norm=w.float().norm().item(), changed_frac=(merged != w).float().mean().item())
                used.add(n); w = merged
            tens[n] = w.contiguous()
    save_file(tens, f"{out}/{shard}", metadata=meta or {"format": "pt"})
    print(f"{shard}: {len(tens)} tensors, merged so far {len(used)}, {time.time()-t0:.0f}s", flush=True)
    del tens
if used != set(ad): stop(f"unmatched adapter modules: {sorted(set(ad)-used)[:5]}")
for fn in os.listdir(base):
    p = f"{base}/{fn}"
    if os.path.isfile(p) and not fn.endswith(".safetensors") and fn != "crc32.txt":
        shutil.copy(p, f"{out}/{fn}")
for fn in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):   # the adapter's own tokenizer files
    if os.path.exists(f"{adir}/{fn}"): shutil.copy(f"{adir}/{fn}", f"{out}/{fn}")
json.dump(dict(adapter_config=cfg, scale=scale, n_modules=len(ad), stats=stats), open(f"{out}/merge_stats.json", "w"), indent=1)
print("merged modules", len(used), "of", len(ad), "adapter modules; done", f"{time.time()-t0:.0f}s")
