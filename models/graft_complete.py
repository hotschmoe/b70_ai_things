#!/usr/bin/env python3
# graft_complete.py -- turn a vision-STRIPPED compressed-tensors build into a COMPLETE one:
# LM (int4/int8 quant) + MTP head (bf16) + vision tower (bf16), one self-contained checkpoint
# with a correct sharded index. CPU-only (no GPU), ~1 min. Mirrors sglang/graft_vision.py +
# sglang/graft_mtp.py, but emits a fully MATERIALIZED dir (no symlinks) under /out.
#
# Why a config splice: the W4A16 base ships architectures=Qwen3_5ForCausalLM with NO
# vision_config, yet its weights already use the multimodal `model.language_model.*` /
# `model.visual.*` layout. So we take the STRUCTURAL config from the bf16 VLM (correct arch +
# vision_config + token ids) and overlay only the quantization_config from the quant build.
#
#   usage: graft_complete.py <IN_mtpgraft_dir> <BF16_dir> <OUT_dir>
#     IN    = the *-mtp-graft dir (LM model.safetensors + model-mtp-graft.safetensors + config)
#     BF16  = the bf16 VLM (source of the 333 visual.* tensors + structural config)
#     OUT   = destination (fully materialized, no links)
import json, os, sys, glob, struct, shutil
from safetensors import safe_open
from safetensors.torch import save_file

IN, BF16, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
BPE = {"I8":1,"U8":1,"BOOL":1,"F8_E4M3":1,"F8_E5M2":1,"BF16":2,"F16":2,"F32":4,"F64":8,"I16":2,"I32":4,"I64":8}

def header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; h=json.loads(f.read(n))
    h.pop("__metadata__", None); return h

# ---- 1. LM base weight: copy the real quant shard(s) ------------------------------------
lm_shards = [f for f in os.listdir(IN) if f.endswith(".safetensors") and "mtp" not in f and "visual" not in f]
assert lm_shards, f"no LM shard in {IN}"
for f in lm_shards:
    shutil.copy2(os.path.join(IN,f), os.path.join(OUT,f))
print(f"LM: copied {len(lm_shards)} shard(s): {lm_shards}")

# ---- 2. MTP head: copy the bf16 mtp graft -> model-mtp.safetensors ----------------------
mtp_src = [f for f in os.listdir(IN) if "mtp" in f and f.endswith(".safetensors")]
assert len(mtp_src)==1, f"expected 1 mtp shard in {IN}, got {mtp_src}"
shutil.copy2(os.path.join(IN,mtp_src[0]), os.path.join(OUT,"model-mtp.safetensors"))
print(f"MTP: {mtp_src[0]} -> model-mtp.safetensors")

# ---- 3. vision tower: graft the 333 bf16 visual.* tensors from the bf16 VLM -------------
bf16_idx = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]
vis_keys = sorted(k for k in bf16_idx if "visual" in k)
assert len(vis_keys)==333, f"expected 333 visual tensors in bf16, got {len(vis_keys)}"
visual={}
for sh in sorted({bf16_idx[k] for k in vis_keys}):
    with safe_open(f"{BF16}/{sh}", framework="pt") as f:
        for k in f.keys():
            if "visual" in k: visual[k]=f.get_tensor(k)
assert len(visual)==333, (len(visual),)
save_file(visual, f"{OUT}/model-visual.safetensors", metadata={"format":"pt"})
print(f"VISION: grafted {len(visual)} visual.* tensors -> model-visual.safetensors")

# ---- 4. sidecar files: tokenizer / preprocessor / chat template (skip backups + config) -
SKIP = (".bak",".ignore",".textonly",".owner")
for f in os.listdir(IN):
    if f.endswith(".safetensors") or f=="config.json" or f.endswith(".index.json"): continue
    if any(x in f for x in SKIP) or f=="MTP_GRAFT_NOTES.txt": continue
    s=os.path.join(IN,f)
    if os.path.isfile(s): shutil.copy2(s, os.path.join(OUT,f))

# ---- 5. config: structural config from bf16 VLM + quant config from the quant build ------
struct_cfg = json.load(open(f"{BF16}/config.json"))
quant_cfg  = json.load(open(f"{IN}/config.json"))
struct_cfg["quantization_config"] = quant_cfg["quantization_config"]
# declare the MTP head where sglang reads it (hf_text_config)
tc = struct_cfg.get("text_config", struct_cfg)
tc["num_nextn_predict_layers"] = 1
json.dump(struct_cfg, open(f"{OUT}/config.json","w"), indent=2)
print(f"CONFIG: arch={struct_cfg.get('architectures')} vision_config={'vision_config' in struct_cfg} "
      f"num_nextn=1 quant={struct_cfg['quantization_config'].get('quant_method')}")

# ---- 6. sharded index: map every tensor in OUT to its shard, sum bytes ------------------
weight_map={}; total=0
for shard in sorted(f for f in os.listdir(OUT) if f.endswith(".safetensors")):
    for k,meta in header(os.path.join(OUT,shard)).items():
        weight_map[k]=shard
        n=1
        for s in meta["shape"]: n*=s
        total += n*BPE[meta["dtype"]]
json.dump({"metadata":{"total_size":total},"weight_map":weight_map},
          open(f"{OUT}/model.safetensors.index.json","w"), indent=1)
nlm=sum(1 for v in weight_map.values() if v in lm_shards)
print(f"INDEX: {len(weight_map)} tensors ({nlm} LM + 15 mtp + 333 visual), total_size={total/1e9:.1f} GB")
print("OK ->", OUT)
