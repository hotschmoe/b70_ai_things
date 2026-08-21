#!/usr/bin/env python3
# Splice the Qwen3.8 VLM wrapper config onto a 151 CausalLM save.
# 151 loaded AutoModelForCausalLM so config.json is qwen3_5_text /
# Qwen3_5ForCausalLM, but shards are model.language_model.* + visual + mtp.
# Same splice as models/graft_qwen38_w8a8.py. Keeps quantization_config.
# Adds visual/mtp to ignore (they were not in the CausalLM, so 151 never
# expanded those regexes). CPU-only. Idempotent if already ConditionalGeneration.
import json
import os
import shutil
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/w4a8-rtn-gdn"
SRC = sys.argv[2] if len(sys.argv) > 2 else \
    "/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/bf16"

cfg_path = os.path.join(OUT, "config.json")
bak = os.path.join(OUT, "config.json.causal_lm_151")
cfg = json.load(open(cfg_path))
qc = cfg.get("quantization_config") or {}
assert qc, "no quantization_config"
if cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"] and "vision_config" in cfg:
    print("already VLM wrapper; skip splice")
    ign = qc.get("ignore") or []
    extra = [p for p in ("re:.*visual.*", "re:.*mtp.*") if p not in ign]
    if extra:
        qc["ignore"] = extra + ign
        cfg["quantization_config"] = qc
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print("added ignore", extra)
    sys.exit(0)

if not os.path.isfile(bak):
    shutil.copy2(cfg_path, bak)
    print("backed up", bak)

struct = json.load(open(os.path.join(SRC, "config.json")))
ign = list(qc.get("ignore") or [])
for p in ("re:.*visual.*", "re:.*mtp.*", "lm_head"):
    if p not in ign:
        ign.insert(0, p)
# vLLM fuses in_proj_qkv+z -> in_proj_qkvz and in_proj_b+a -> in_proj_ba.
# 151 regexes are the unfused HF names; the fused Linear would otherwise
# fall through to W4A8 packed-int4 and shape-assert on I8/BF16 shards.
for gn, g in (qc.get("config_groups") or {}).items():
    w = g.get("weights") or {}
    if w.get("num_bits") == 8 and w.get("strategy") == "channel":
        t = list(g.get("targets") or [])
        extra = "re:.*linear_attn\\.in_proj_qkvz$"
        if extra not in t:
            t.append(extra)
        g["targets"] = t
for ptn in ("re:.*linear_attn\\.in_proj_ba$",):
    if ptn not in ign:
        ign.insert(0, ptn)
qc["ignore"] = ign
struct["quantization_config"] = qc
struct["architectures"] = ["Qwen3_5ForConditionalGeneration"]
struct["model_type"] = "qwen3_5"
struct["language_model_only"] = False
tc = struct.setdefault("text_config", {})
tc["num_nextn_predict_layers"] = 1
json.dump(struct, open(cfg_path, "w"), indent=2)
print("CONFIG", struct["architectures"], "vision_config", "vision_config" in struct,
      "is_prepacked", qc.get("is_prepacked_w4a8"), "n_ignore", len(ign))

for name in ("preprocessor_config.json", "video_preprocessor_config.json",
             "processor_config.json"):
    src = os.path.join(SRC, name)
    dst = os.path.join(OUT, name)
    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print("copied", name)
