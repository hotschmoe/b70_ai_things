#!/usr/bin/env python3
# Graft official Qwen3.8-27B BF16 vision + MTP onto the on-box W8A8-gptq dir.
# CPU-only. Writes model-visual.safetensors + model-mtp.safetensors in place,
# splices the VLM wrapper config, rebuilds the shard index.
#
#   python graft_qwen38_w8a8.py <W8A8_DIR> <BF16_DIR>
import json
import os
import struct
import sys

from safetensors import safe_open
from safetensors.torch import save_file

W8, BF16 = sys.argv[1], sys.argv[2]
assert os.path.isdir(W8) and os.path.isdir(BF16)

BPE = {
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I16": 2, "I32": 4, "I64": 8,
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


bf16_idx = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))[
    "weight_map"
]
vis_keys = sorted(k for k in bf16_idx if "visual" in k)
mtp_keys = sorted(k for k in bf16_idx if k.startswith("mtp."))
assert len(vis_keys) == 333, len(vis_keys)
assert len(mtp_keys) == 15, mtp_keys

visual = {}
for sh in sorted({bf16_idx[k] for k in vis_keys}):
    with safe_open(os.path.join(BF16, sh), framework="pt") as f:
        for k in f.keys():
            if "visual" in k:
                visual[k] = f.get_tensor(k)
assert len(visual) == 333, len(visual)
vis_path = os.path.join(W8, "model-visual.safetensors")
save_file(visual, vis_path, metadata={"format": "pt"})
print("VISION", len(visual), "->", vis_path)

mtp = {}
for sh in sorted({bf16_idx[k] for k in mtp_keys}):
    with safe_open(os.path.join(BF16, sh), framework="pt") as f:
        for k in f.keys():
            if k.startswith("mtp."):
                mtp[k] = f.get_tensor(k)
assert len(mtp) == 15, sorted(mtp)
mtp_path = os.path.join(W8, "model-mtp.safetensors")
save_file(mtp, mtp_path, metadata={"format": "pt"})
print("MTP", sorted(mtp), "->", mtp_path)

struct_cfg = json.load(open(os.path.join(BF16, "config.json")))
quant_cfg = json.load(open(os.path.join(W8, "config.json")))
qc = quant_cfg["quantization_config"]
# keep ignore covering visual/mtp so vLLM does not try to INT8 them
struct_cfg["quantization_config"] = qc
struct_cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
struct_cfg["model_type"] = "qwen3_5"
struct_cfg["language_model_only"] = False
tc = struct_cfg.setdefault("text_config", {})
tc["num_nextn_predict_layers"] = 1
json.dump(struct_cfg, open(os.path.join(W8, "config.json"), "w"), indent=2)
print(
    "CONFIG",
    struct_cfg["architectures"],
    "vision_config",
    "vision_config" in struct_cfg,
    "num_nextn",
    tc.get("num_nextn_predict_layers"),
)

for name in (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
):
    src = os.path.join(BF16, name)
    dst = os.path.join(W8, name)
    if os.path.isfile(src) and not os.path.isfile(dst):
        open(dst, "wb").write(open(src, "rb").read())
        print("copied", name)

weight_map = {}
total = 0
for shard in sorted(f for f in os.listdir(W8) if f.endswith(".safetensors")):
    for k, meta in header(os.path.join(W8, shard)).items():
        weight_map[k] = shard
        n = 1
        for s in meta["shape"]:
            n *= s
        total += n * BPE[meta["dtype"]]
json.dump(
    {"metadata": {"total_size": total}, "weight_map": weight_map},
    open(os.path.join(W8, "model.safetensors.index.json"), "w"),
    indent=1,
)
n_vis = sum(1 for k in weight_map if "visual" in k)
n_mtp = sum(1 for k in weight_map if k.startswith("mtp."))
print(
    "INDEX",
    len(weight_map),
    "visual",
    n_vis,
    "mtp",
    n_mtp,
    "total_size_gb",
    round(total / 1e9, 2),
)
assert n_vis == 333 and n_mtp == 15
print("OK", W8)
