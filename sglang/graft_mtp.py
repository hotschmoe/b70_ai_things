#!/usr/bin/env python3
# graft_mtp.py -- graft the BF16 MTP (NEXTN) head from an mtp-graft checkpoint onto the Lorbus int4
# (auto-round) checkpoint, so sglang can serve int4 (woqgemm) + vision + MTP speculative decode in one
# checkpoint. The base Qwen3.6-27B ships NO MTP head; the head was grafted (15 BF16 mtp.* weights:
# fc + 1 transformer layer + norms). Lorbus already has int4-LM + vision; we only add the MTP head.
# CPU-only: symlinks the LM shards, adds one mtp shard, updates the index + config. ~seconds.
#   usage: python3 graft_mtp.py [SRC_LM] [SRC_MTP] [DST]
import json, os, sys, glob
from safetensors import safe_open
from safetensors.torch import save_file

SRC_LM = sys.argv[1] if len(sys.argv) > 1 else "/models/Lorbus_Qwen3.6-27B-int4-AutoRound"
SRC_MTP = sys.argv[2] if len(sys.argv) > 2 else "/models/Qwen3.6-27B-W4A16-mtp-graft"
DST = sys.argv[3] if len(sys.argv) > 3 else "/models/Lorbus_Qwen3.6-27B-int4-mtp"
os.makedirs(DST, exist_ok=True)

# 1. symlink every LM file except the config + index (we rewrite those)
for f in os.listdir(SRC_LM):
    if f in ("config.json", "model.safetensors.index.json"):
        continue
    s, d = os.path.join(SRC_LM, f), os.path.join(DST, f)
    if os.path.isfile(s) and not os.path.lexists(d):
        os.symlink(s, d)

# 2. extract the BF16 mtp.* head weights -> a single shard in DST
mtp = {}
for f in sorted(glob.glob(SRC_MTP + "/*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("mtp.") or "nextn" in k:
                mtp[k] = h.get_tensor(k)
save_file(mtp, os.path.join(DST, "model-mtp.safetensors"), metadata={"format": "pt"})

# 3. update the safetensors index to point the mtp.* tensors at the new shard
idx = json.load(open(os.path.join(SRC_LM, "model.safetensors.index.json")))
for k in mtp:
    idx["weight_map"][k] = "model-mtp.safetensors"
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"))

# 4. declare the MTP head: sglang reads num_nextn_predict_layers from hf_text_config (model_config.py:929)
cfg = json.load(open(os.path.join(SRC_LM, "config.json")))
tc = cfg.get("text_config", cfg)
tc["num_nextn_predict_layers"] = 1
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print(f"GRAFT DONE: {len(mtp)} mtp weights -> {DST}; num_nextn_predict_layers=1")
print("Serve: --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 "
      "--speculative-num-draft-tokens 2 --speculative-draft-attention-backend triton (dodges XPU gates)")
