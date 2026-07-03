#!/usr/bin/env python3
# quant_dflash_drafter.py -- DATA-FREE RTN W8A8 int8 quant of the DFlash drafter.
#
# WHY RTN (not SmoothQuant+GPTQ like our 27B target checkpoints): calibrated GPTQ is
# INFEASIBLE for this drafter, for two independent reasons (see vllm/DFLASH_FOLLOWUP_PREP.md):
#   1. No HF modeling class. config.json has architectures=["DFlashDraftModel"] +
#      auto_map AutoModel->"dflash.DFlashDraftModel", but the checkpoint dir ships NO dflash.py.
#      So transformers/llmcompressor cannot instantiate the model (trust_remote_code has nothing
#      to import; model_type "qwen3" does not match the DFlash state dict: extra fc/hidden_norm).
#   2. The drafter's real input is TARGET hidden states. Its forward combines hidden states
#      tapped from 5 target layers (fc: Linear[5120, 25600] = 5*5120) via combine_hidden_states.
#      Pure-text calibration THROUGH the drafter alone is impossible without co-running the 27B
#      target and plumbing per-layer hidden states -- which llmcompressor's SequentialPipeline
#      cannot do. So there is no clean calibration forward pass.
#
# RTN needs NO calibration forward and NO modeling class: it operates directly on the raw
# safetensors weights. Output is compressed-tensors "int-quantized" W8A8 (per-channel symmetric
# int8 weights + dynamic per-token int8 activations) -- byte-format-identical to our
# w8a8-sqgptq target checkpoints, so vLLM's compressed-tensors loader + the B70 int8 XMX path
# consume it unchanged. Quality is lower than GPTQ (no outlier smoothing) but the drafter only
# needs to PROPOSE tokens; the W8A8 TARGET verifies every token, so drafter RTN error costs a
# little accept-length, never correctness (spec decoding is lossless w.r.t. the target).
#
# RUN: CPU-only container, NO --device, NO GPU lease. Example (orchestrator):
#   docker run --rm --entrypoint bash \
#     -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
#     -v /mnt/vm_8tb/github/b70_ai_things:/repo -w /repo \
#     vllm-xpu-env:int8g-v0240 -c 'python3 vllm/quant_dflash_drafter.py'
# Needs only torch + safetensors (both in the image). ~3.3GB in -> ~1.9GB out, a few minutes.
#
# Env knobs:
#   SRC     (default models/files/qwen3.6-27b/dflash-draft)
#   OUT     (default models/files/qwen3.6-27b/dflash-draft-w8a8-rtn)
#           NOTE: tagged -rtn NOT -sqgptq on purpose. Per CLAUDE.md Model Identity, output dirs
#           must encode the REAL method. This is RTN, not SmoothQuant+GPTQ. Do not rename to
#           -sqgptq (that mislabel is exactly the corruption the repo rule guards against).
#   QUANT_FC (default 0) -- keep the fc target-hidden-state adapter in bf16 by default
#           (it is the accuracy-critical input projection). Set 1 to also int8-quantize fc.
#
# ASCII only.

import json
import os
import shutil
import struct
import sys

import torch
from safetensors.torch import save_file

SRC = os.environ.get("SRC", "models/files/qwen3.6-27b/dflash-draft")
OUT = os.environ.get("OUT", "models/files/qwen3.6-27b/dflash-draft-w8a8-rtn")
QUANT_FC = os.environ.get("QUANT_FC", "0").strip() not in ("0", "", "false", "no", "off")

# Linear weights we int8-quantize. Norms (q_norm/k_norm/*_layernorm/hidden_norm/norm) stay bf16.
# q/k/v MUST all stay bf16 (2026-07-03 serve tests, two walls):
# (1) int8 k/v: vLLM's qwen3_dflash.py precompute_and_store_context_kv builds _fused_kv_weight
#     from the RAW k/v weights and calls F.linear directly (bypasses the quantized wrapper) ->
#     "expected mat1 and mat2 to have the same dtype: BFloat16 != signed char" at init.
# (2) int8 q with bf16 k/v: vLLM fuses q/k/v into ONE qkv_proj module which must be uniformly
#     quantized; mixed shards -> module built unquantized -> loader KeyError
#     'layers.0.self_attn.qkv_proj.weight_scale'.
# So attention QKV is bf16 (~315 MB across 5 layers); int8 = o_proj + MLP (the byte bulk).
PROJ_SUFFIXES = (
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)


def is_quant_target(name):
    if name == "fc.weight":
        return QUANT_FC
    return any(name.endswith(s) for s in PROJ_SUFFIXES)


def load_safetensors_headered(path):
    """Read all tensors from a single-file safetensors into a name->tensor dict."""
    from safetensors.torch import load_file
    return load_file(path)


def rtn_int8_per_channel(w):
    """Per-output-channel symmetric int8 RTN. Returns (int8 weight [out,in], bf16 scale [out,1]).

    Matches compressed-tensors 'channel' + symmetric int-quantized: scale = amax(row)/127,
    q = round-to-nearest-even(w/scale) clamped to [-128, 127].
    """
    w32 = w.to(torch.float32)
    amax = w32.abs().amax(dim=1, keepdim=True)          # [out, 1]
    amax = torch.clamp(amax, min=1e-8)
    scale = amax / 127.0                                 # [out, 1]
    q = torch.round(w32 / scale)                         # round half to even (torch default)
    q = torch.clamp(q, -128, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def build_quantization_config(ignored_linears):
    return {
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "input_activations": {
                    "actorder": None, "block_structure": None, "dynamic": True,
                    "group_size": None, "num_bits": 8, "observer": None,
                    "observer_kwargs": {}, "scale_dtype": None, "strategy": "token",
                    "symmetric": True, "type": "int", "zp_dtype": None,
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None, "block_structure": None, "dynamic": False,
                    "group_size": None, "num_bits": 8, "observer": "memoryless_minmax",
                    "observer_kwargs": {}, "scale_dtype": None, "strategy": "channel",
                    "symmetric": True, "type": "int", "zp_dtype": None,
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": None,
        "ignore": ignored_linears,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.17.1",
    }


def main():
    src_st = os.path.join(SRC, "model.safetensors")
    if not os.path.isfile(src_st):
        print("MISSING drafter safetensors: %s" % src_st, file=sys.stderr)
        return 1
    os.makedirs(OUT, exist_ok=True)

    print("[load] %s" % src_st, flush=True)
    tensors = load_safetensors_headered(src_st)

    out_tensors = {}
    n_q = n_keep = 0
    quantized_bytes = 0
    for name, t in tensors.items():
        if is_quant_target(name):
            q, scale = rtn_int8_per_channel(t)
            out_tensors[name] = q.contiguous()
            out_tensors[name[:-len(".weight")] + ".weight_scale"] = scale.contiguous()
            n_q += 1
            quantized_bytes += q.numel()
        else:
            out_tensors[name] = t.contiguous()
            n_keep += 1
    print("[quant] int8 linears=%d  kept-bf16 tensors=%d  fc_quantized=%s"
          % (n_q, n_keep, QUANT_FC), flush=True)

    out_st = os.path.join(OUT, "model.safetensors")
    save_file(out_tensors, out_st, metadata={"format": "pt"})
    print("[save] %s (%.2f GB)" % (out_st, os.path.getsize(out_st) / 1e9), flush=True)

    # ---- config.json: preserve dflash_config/block_size/auto_map/arch, add quantization_config ----
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    # Which Linear modules stay bf16 -> must be listed in the compressed-tensors ignore list.
    # (Norms are not Linear, so they need not be listed.) fc is a Linear; ignore it unless quantized.
    ignored = ["re:.*q_proj.*", "re:.*k_proj.*", "re:.*v_proj.*"]  # see PROJ_SUFFIXES note
    if not QUANT_FC:
        ignored.append("re:.*fc.*")
    cfg["quantization_config"] = build_quantization_config(ignored)
    # Sanity: keep the DFlash-critical fields intact (they gate the vLLM drafter arch).
    assert cfg.get("architectures") == ["DFlashDraftModel"], "lost architectures"
    assert "dflash_config" in cfg and "block_size" in cfg, "lost dflash_config/block_size"
    assert cfg.get("auto_map"), "lost auto_map"
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    print("[config] wrote quantization_config, preserved dflash_config/block_size/auto_map", flush=True)

    # README is nice-to-have provenance; assets/tokenizer are not needed (drafter shares the
    # target tokenizer + embedding/LM-head). Copy README only if present.
    for extra in ("README.md",):
        s = os.path.join(SRC, extra)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, extra))

    print("DONE_DFLASH_W8A8_RTN %s" % OUT, flush=True)
    print("[verdict] RTN weight-only int8; no calibration was possible (see header). The W8A8 "
          "target still verifies every token, so this is lossless w.r.t. the target -- it only "
          "trades a little accept-length for a halved drafter weight read.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
