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
ocfg["quantization_config"] = qcfg["quantization_config"]
json.dump(ocfg, open(os.path.join(QUANT, "config.json"), "w"), indent=2)
print("[fix] arch:", ocfg.get("architectures"), "| model_type:", ocfg.get("model_type"),
      "| has text_config:", "text_config" in ocfg,
      "| quant_method:", ocfg["quantization_config"].get("quant_method"),
      "| ignore:", ocfg["quantization_config"].get("ignore"))
