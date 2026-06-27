#!/usr/bin/env python3
# graft_vision.py -- give the text-only W8A8 ckpt its vision tower back WITHOUT a GPU requant.
# The W8A8-sqgptq quant excluded `re:.*visual.*` (vision stays bf16) but the quantizer also DROPPED
# the 333 visual.* tensors from the output (model.safetensors has 0 visual). Since they're meant to be
# bf16-unquantized, the source bf16 VLM's visual weights ARE byte-correct -> graft them in (like the
# MTP graft). Produces a sharded ckpt: symlinked int8 base + a new bf16 visual shard + an index.
#
# Run as ROOT inside a container with models mounted rw:
#   docker run --rm -u 0 -v /mnt/vm_8tb/b70/models:/models sglang-xpu:woq \
#     python /graft.py   (mount this file to /graft.py)
import json, os, struct
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/Qwen_Qwen3.6-27B"                       # bf16 VLM (333 visual tensors)
W8  = "/models/Qwen3.6-27B-W8A8-sqgptq"                # int8 LM, 0 visual
OUT = "/models/Qwen3.6-27B-W8A8-sqgptq-vision"
W8_BASENAME = os.path.basename(W8)
os.makedirs(OUT, exist_ok=True)

# 1. collect the bf16 visual.* tensors from the source shards
src_idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
vis_keys = sorted(k for k in src_idx if "visual" in k)
shards = sorted({src_idx[k] for k in vis_keys})
visual = {}
for sh in shards:
    with safe_open(f"{SRC}/{sh}", framework="pt") as f:
        for k in f.keys():
            if "visual" in k:
                visual[k] = f.get_tensor(k)
assert len(visual) == len(vis_keys) == 333, (len(visual), len(vis_keys))
save_file(visual, f"{OUT}/model-visual.safetensors", metadata={"format": "pt"})
print(f"wrote {len(visual)} visual tensors -> {OUT}/model-visual.safetensors")

# 2. read existing int8 base keys + dtype/shape (for the index byte-size totals)
with open(f"{W8}/model.safetensors", "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
base_keys = [k for k in hdr if k != "__metadata__"]

# 3. symlink the int8 base weight + every non-weight sidecar file (relative -> resolves under /models)
def link(name, rel_tgt):
    p = f"{OUT}/{name}"
    if os.path.lexists(p):
        os.remove(p)
    os.symlink(rel_tgt, p)

link("model.safetensors", f"../{W8_BASENAME}/model.safetensors")
for fn in os.listdir(W8):
    if fn.endswith(".safetensors") or fn.endswith(".bak"):
        continue                                  # skip the base weight (linked above) + config backups
    if fn.startswith("config.json") and fn != "config.json":
        continue
    link(fn, f"../{W8_BASENAME}/{fn}")

# 4. build the sharded index (every tensor -> its file) + total_size
BPE = {"I8": 1, "U8": 1, "BOOL": 1, "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I16": 2, "I32": 4, "I64": 8}
def nbytes(meta):
    n = 1
    for s in meta["shape"]:
        n *= s
    return n * BPE[meta["dtype"]]

weight_map = {k: "model.safetensors" for k in base_keys}
for k in visual:
    weight_map[k] = "model-visual.safetensors"
total = sum(nbytes(hdr[k]) for k in base_keys) + sum(t.numel() * t.element_size() for t in visual.values())
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(f"{OUT}/model.safetensors.index.json", "w"), indent=1)
print(f"index: {len(base_keys)} int8/bf16 base + {len(visual)} visual = {len(weight_map)} tensors, total_size={total}")
print("OK ->", OUT)
