#!/usr/bin/env python3
# Fix the collapsed config of a compressed-tensors-quantized Qwen3.6-27B (VLM). llmcompressor +
# AutoModelForCausalLM save the TEXT config (model_type qwen3_5_text / Qwen3_5ForCausalLM), which
# vLLM-XPU rejects (it wants the Qwen3_5Config wrapper). The QUANTIZED WEIGHTS are correctly named
# (model.language_model.* + model.visual.*), so we just restore the original VLM wrapper config and
# inject the quantization_config. Env: ORIG (bf16 source config dir), QUANT (quantized dir to fix).
import json, os, shutil
ORIG = os.environ["ORIG"]; QUANT = os.environ["QUANT"]
qcfg = json.load(open(os.path.join(QUANT, "config.json")))
ocfg = json.load(open(os.path.join(ORIG, "config.json")))
assert "quantization_config" in qcfg, "quantized config has no quantization_config -- nothing to graft"
shutil.copy(os.path.join(QUANT, "config.json"), os.path.join(QUANT, "config.json.textonly.bak"))
qc = qcfg["quantization_config"]
# [!] llmcompressor loads the qwen3_5 VLM via AutoModelForCausalLM (text-only fallback) so it NEVER sees
# the vision tower / MTP head -> the saved `ignore` list lacks them. When we graft the VLM wrapper config
# back (which re-introduces model.visual.* + mtp.*), vLLM matches those Linears to the config_group
# (targets=["Linear"]) and tries to int4-quantize the 4304-dim vision fc2 -> "input_size_per_partition
# 4304 not divisible by group_size 128" assert at serve. Fix: ensure the regex ignores are present so
# vLLM keeps vision/MTP in bf16. (W8A8 per-channel would not assert but should still keep vision bf16.)
ig = list(qc.get("ignore", []))
for pat in ["re:.*visual.*", "re:.*mtp.*"]:
    if pat not in ig:
        ig.append(pat)
qc["ignore"] = ig
ocfg["quantization_config"] = qc
json.dump(ocfg, open(os.path.join(QUANT, "config.json"), "w"), indent=2)
print("[fix] arch:", ocfg.get("architectures"), "| model_type:", ocfg.get("model_type"),
      "| has text_config:", "text_config" in ocfg,
      "| quant_method:", qc.get("quant_method"), "| ignore_count:", len(ig),
      "| visual_ignored:", any("visual" in x for x in ig), "| mtp_ignored:", any("mtp" in x for x in ig))
