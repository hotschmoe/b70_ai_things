#!/usr/bin/env bash
# Track C (W4A4 rotation) Phase C1 Step 1: Qwen3-14B W4A4 QuaRot-style quant.
# Copy of scripts/145_w4a4_smoke_0p6b.sh generalized to 14B (dense, UNTIED embeddings).
# Recipe: SpinQuantModifier(R1,R2,R4 random-hadamard) + GPTQModifier custom W4A4 scheme
# (weights int4 g128 sym; input_activations int4 per-token dynamic sym). See 145's header
# for the three Step-0 mechanics fixes (online-R4 reapply at load, tied-lm_head untie,
# _tied_weights_keys=None at save) -- all kept here as guards (no-ops for untied 14B).
#
# Env:
#   DEVICE   cpu (default) | xpu   -- xpu adds --device /dev/dri + ZE_AFFINITY_MASK=$CARD.
#            xpu runs MUST go through bin/gpu-run (see AGENTS.md GPU discipline).
#   CARD     1 (default; only used when DEVICE=xpu)
#   ROTATE   1 (default) | 0 = skip SpinQuantModifier (norot ablation); OUT gets -norot
#   SAMPLES (128) SEQLEN (2048) SRC (/specula_models/Qwen3-14B)
#   OUT      default /mnt/vm_8tb/b70/quant_work/Qwen3-14B-W4A4-quarot-gptq[-norot]
#   SKIP_QUANT=1  skip Phase A, reuse existing OUT (Phase B iteration)
#   OMP      OMP_NUM_THREADS in container (default 24; DD shares the box CPUs)
#
# CPU run is multi-hour (0.6B took ~8 min at 1/23 params and 1/32 calib tokens).
# --cpu-shares 512 + OMP=24 keep the DD serving responsive; RAM capped at 90g.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
IMG="${IMG:-vllm-xpu-env:int8g-v0251}"
SRC="${SRC:-/specula_models/Qwen3-14B}"
DEVICE="${DEVICE:-cpu}"; CARD="${CARD:-1}"
ROTATE="${ROTATE:-1}"
SAMPLES="${SAMPLES:-128}"; SEQLEN="${SEQLEN:-2048}"
SUFFIX=""; [ "$ROTATE" = 0 ] && SUFFIX="-norot"
OUT="${OUT:-$ROOT/quant_work/Qwen3-14B-W4A4-quarot-gptq$SUFFIX}"
LOG="$ROOT/results/quant14b_w4a4${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/quant_work" "$ROOT/pip_cache"

GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" --cpu-shares 512)
[ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK="$CARD")
echo "=== 14B W4A4 quant: device=$DEVICE rotate=$ROTATE samples=$SAMPLES seqlen=$SEQLEN out=$OUT ==="
echo "=== log: $LOG ==="

docker run --rm --name "quant14b_w4a4${SUFFIX}" "${GPUARGS[@]}" \
  --ipc=host --shm-size 16g --memory 90g \
  -v "$ROOT:$ROOT" -v "$SPECULA:/specula_models:ro" \
  -e HF_HOME="$ROOT/hf_cache" -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e OMP_NUM_THREADS="${OMP:-24}" -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e OUT="$OUT" -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" \
  -e ROTATE="$ROTATE" -e SKIP_QUANT="${SKIP_QUANT:-0}" \
  --entrypoint bash "$IMG" -c '
    set -e
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -2 || true
    pip list 2>/dev/null | grep -Ei "^(llmcompressor|compressed-tensors|transformers|torch|accelerate|datasets) " || true

    if [ "${SKIP_QUANT:-0}" = 1 ]; then echo "=== PHASE A SKIPPED (SKIP_QUANT=1) ==="; else
    echo "=== PHASE A: SpinQuant(R1,R2,R4)[ROTATE-gated] + GPTQ W4A4 oneshot + save_compressed ==="
    python - <<PY
import os, time, torch
t_start=time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

SRC=os.environ["SRC"]; OUT=os.environ["OUT"]
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"])
ROTATE=os.environ.get("ROTATE","1")=="1"

print(f"[load] {SRC} bf16 on CPU ...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu",
                                           low_cpu_mem_usage=True, trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

# Guard from Step 0 (scripts/145): under transformers 5.x, ct untie_word_embeddings only
# flips the config flag -- tensors stay shared and save drops lm_head. 14B is untied so
# this no-ops, but keep the guard for any tied model.
if getattr(model.config, "tie_word_embeddings", False):
    import torch.nn as nn
    emb=model.get_input_embeddings(); head=model.get_output_embeddings()
    if head.weight.data_ptr()==emb.weight.data_ptr():
        head.weight=nn.Parameter(emb.weight.detach().clone())
    model.config.tie_word_embeddings=False
    print(f"[fix] untied lm_head from embed_tokens (shared={head.weight.data_ptr()==emb.weight.data_ptr()})", flush=True)
t_load=time.time(); print(f"[time] load {t_load-t_start:.0f}s", flush=True)

from datasets import load_dataset
ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
ds=ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
ds=ds.map(lambda s: tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False),
          remove_columns=ds.column_names)
t_data=time.time(); print(f"[time] dataset {t_data-t_load:.0f}s", flush=True)

# Custom W4A4 scheme: no compressed-tensors preset -- built by hand.
w4a4=QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                             strategy="group", group_size=128),
    input_activations=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                                       strategy="token", dynamic=True),
)
q=GPTQModifier(config_groups={"group_0": w4a4}, ignore=["lm_head"], actorder=None)
recipe=([SpinQuantModifier(rotations=["R1","R2","R4"], transform_type="random-hadamard"), q]
        if ROTATE else [q])
tag="SpinQuant(R1,R2,R4)+GPTQ" if ROTATE else "GPTQ-only (NOROT ablation)"
print(f"[quant] {tag} W4A4 g128/token-dyn, N={N} seq={SEQ} ...", flush=True)
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
t_q=time.time(); print(f"[time] oneshot {t_q-t_data:.0f}s", flush=True)

# Guard from Step 0: transformers 5.x save_pretrained drops _tied_weights_keys even
# after untying; clear the marker so lm_head serializes (no-op when nothing shared).
model._tied_weights_keys = None
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
print(f"[time] save {time.time()-t_q:.0f}s", flush=True)
import glob, json as _json, struct
saved=set()
for f in glob.glob(OUT+"/*.safetensors"):
    with open(f,"rb") as fh:
        n=struct.unpack("<Q", fh.read(8))[0]; saved |= set(_json.loads(fh.read(n)))
has_head="lm_head.weight" in saved  # NOTE: no nested single-quotes -- outer bash -c is single-quoted
print(f"[verify] lm_head.weight in checkpoint: {has_head}", flush=True)
print("DONE_W4A4_14B_QUANT", OUT, flush=True)
PY
    fi

    echo "=== saved config.json quantization_config (transform_config nested inside) ==="
    python - <<PY
import json, os
cfg=json.load(open(os.environ["OUT"]+"/config.json"))
qc=cfg.get("quantization_config","<ABSENT>")
print("tie_word_embeddings:", cfg.get("tie_word_embeddings"))
print(json.dumps(qc, indent=2)[:6000])
PY

    echo "=== PHASE B: fresh-process reload (QDQ run_compressed=False) + greedy generate ==="
    python - <<PY
import os, time, torch
t0=time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig
OUT=os.environ["OUT"]
model=AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16, device_map="cpu",
        quantization_config=CompressedTensorsConfig(run_compressed=False), trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(OUT, trust_remote_code=True)
print(f"[time] reload {time.time()-t0:.0f}s", flush=True)

def count_hooks():
    return sum(1 for name,m in model.named_modules()
               if "down_proj" in name and (m._forward_pre_hooks or m._forward_hooks))
print(f"[diag] down_proj hooks after plain load: {count_hooks()}", flush=True)

# transformers does NOT apply the saved transform_config (only llmcompressor oneshot
# and vLLM do). Re-apply the ONLINE-only entries (location input/output); the
# weight-location entries are already fused into the saved weights -- re-applying
# them would double-rotate. No-op for the -norot ablation (no transform_config).
import json
tcfg_dict=json.load(open(OUT+"/config.json"))["quantization_config"].get("transform_config")
if tcfg_dict:
    from compressed_tensors.transform import (TransformConfig, TransformScheme,
                                              apply_transform_config)
    online_groups={}
    for gname,scheme in TransformConfig.model_validate(tcfg_dict).config_groups.items():
        online=[a for a in scheme.apply if str(a.location) in ("input","output")]
        if online:
            s=scheme.model_copy(); s.apply=online
            online_groups[gname]=s
    if online_groups:
        print(f"[fix] applying ONLINE transforms: "
              f"{ {g:[ (a.location,a.targets) for a in s.apply] for g,s in online_groups.items()} }", flush=True)
        apply_transform_config(model, TransformConfig(config_groups=online_groups))
print(f"[diag] down_proj hooks after online-transform apply: {count_hooks()}", flush=True)

for label,prompt in [("france","The capital of France is"),
                     ("code","def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n")]:
    ids=tok(prompt, return_tensors="pt").input_ids
    t1=time.time()
    out=model.generate(ids, max_new_tokens=60, do_sample=False)
    txt=tok.decode(out[0], skip_special_tokens=True)
    print(f"=== GEN[{label}] ({time.time()-t1:.0f}s) ===\n{txt}\n=== END GEN[{label}] ===", flush=True)
print("DONE_W4A4_14B_GEN", flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="; du -sh "$OUT" 2>/dev/null
