"""Repair an AutoRound-on-VLM checkpoint's config.json so it SERVES on vLLM (the serve-side counterpart to the
MLLM-calib dodge in scripts/84). Run on CPU, no GPU.

THE PROBLEM (Q8, 2026-06-22): AutoRound MLLM-mode on the qwen3_5 VLM (Qwable/Lorbus arch) saves a checkpoint whose
weights use the multimodal naming `model.language_model.layers.*` + `model.visual.*` + `mtp.*`, BUT:
  1. config.json is a FLAT `qwen3_5_text` config: architectures=[Qwen3_5ForCausalLM], model_type=qwen3_5_text,
     NO vision_config / text_config. vLLM still routes to the VLM class (qwen3_5.py) -> __init__ does
     `config.vision_config` -> AttributeError: 'Qwen3_5TextConfig' object has no attribute 'vision_config'.
  2. quantization_config.extra_config (the per-layer bf16 overrides for the DeltaNet linear_attn) is named
     `model.layers.N.linear_attn.*` -- WRONG prefix; the served layers are `model.language_model.layers.N.*`.
     -> vLLM can't find the bf16 override -> tries to int4-dispatch the bf16 linear_attn weights -> load error.

THE FIX (matches the PROVEN Lorbus 27B int4-AutoRound config structure):
  served config.json = BASE model's multimodal config (architectures=Qwen3_5ForConditionalGeneration, model_type=
  qwen3_5, vision_config+text_config) + the quantized checkpoint's `quantization_config` with extra_config keys
  renamed `model.layers.` -> `model.language_model.layers.`. The int4/bf16 split is ground-truth-consistent with the
  weights (verified via the .qweight/.weight tensors in model.safetensors.index.json), so no re-quant is needed.

Usage: python3 87_fix_autoround_vlm_config.py BASE_CONFIG QUANT_CONFIG OUT_CONFIG
  BASE_CONFIG  = original (bf16) model's config.json (the multimodal wrapper for THIS exact model)
  QUANT_CONFIG = the AutoRound output's config.json (has the flat text wrapper + quantization_config)
  OUT_CONFIG   = where to write the repaired config (install it over the quant checkpoint's config.json; back up first)
"""
import json, sys

def repair(base_path, quant_path, out_path):
    base = json.load(open(base_path))
    quant = json.load(open(quant_path))
    qc = quant.get("quantization_config")
    if qc is None:
        sys.exit("quant config.json has no quantization_config -- is this an AutoRound output?")
    ec = qc.get("extra_config", {})
    renamed = 0; new_ec = {}
    for k, v in ec.items():
        nk = k.replace("model.layers.", "model.language_model.layers.")
        if nk != k:
            renamed += 1
        new_ec[nk] = v
    qc["extra_config"] = new_ec
    if "vision_config" not in base or "architectures" not in base:
        sys.exit(f"BASE config {base_path} is not a multimodal wrapper (no vision_config/architectures)")
    base["quantization_config"] = qc
    json.dump(base, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}: arch={base['architectures']} model_type={base.get('model_type')} "
          f"vision_config={'vision_config' in base} extra_config={len(new_ec)} renamed={renamed}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    repair(sys.argv[1], sys.argv[2], sys.argv[3])
