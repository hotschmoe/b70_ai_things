#!/usr/bin/env bash
# Track C (W4A4 rotation) Phase C1 Step 0: CPU-ONLY pipeline smoke on Qwen3-0.6B.
# QuaRot-style recipe via llmcompressor: SpinQuantModifier(R1,R2,R4 random-hadamard)
# + GPTQModifier with a CUSTOM W4A4 scheme (no compressed-tensors preset exists):
#   weights: int4 group-128 symmetric; input_activations: int4 per-token dynamic symmetric.
# Smoke-tests MECHANICS only (mapping fallback on Qwen3, R4 online hook, serialization
# round-trip, reload+generate). A 0.6B W4A4 will be rough -- coherence bar is LOW.
#
# CPU + network ONLY. NO GPU: no --device flag, ZE_AFFINITY_MASK forced empty.
# Mirrors scripts/54_quantize_14b.sh container style (throwaway pip-install container).
#
# Env:
#   SAMPLES (16) SEQLEN (512) MODEL (Qwen/Qwen3-0.6B)
#   OUT     default /mnt/vm_8tb/b70/quant_work/Qwen3-0.6B-W4A4-quarot-smoke
#   SKIP_QUANT=1  skip Phase A, reuse existing OUT (Phase B iteration)
#
# FINDING (run 1): transformers' compressed-tensors loader does NOT apply the saved
# transform_config at load (only llmcompressor oneshot and vLLM's ct-transform support
# do), so the online R4 input rotation was missing on reload -> token salad. Phase B
# now re-applies the ONLINE-only transform entries (location input/output) manually;
# the weight-location entries are already fused into the saved weights (do NOT re-apply).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8g-v0251}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SAMPLES="${SAMPLES:-16}"; SEQLEN="${SEQLEN:-512}"
OUT="${OUT:-$ROOT/quant_work/Qwen3-0.6B-W4A4-quarot-smoke}"
LOG="$ROOT/results/w4a4_smoke_0p6b_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/quant_work" "$ROOT/pip_cache"

echo "=== W4A4 quarot smoke: model=$MODEL samples=$SAMPLES seqlen=$SEQLEN out=$OUT ==="
echo "=== log: $LOG ==="

# CPU-ONLY: no --device, GPU env vars explicitly blanked (mirrors 54's DEVICE=cpu branch).
docker run --rm --name w4a4_smoke \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
  --ipc=host --shm-size 16g --memory 90g \
  -v "$ROOT:$ROOT" \
  -e HF_HOME="$ROOT/hf_cache" -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e OMP_NUM_THREADS=32 -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e MODEL="$MODEL" -e OUT="$OUT" -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" \
  --entrypoint bash "$IMG" -c '
    set -e
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -2 || true
    pip list 2>/dev/null | grep -Ei "^(llmcompressor|compressed-tensors|transformers|torch|accelerate|datasets) " || true

    if [ "${SKIP_QUANT:-0}" = 1 ]; then echo "=== PHASE A SKIPPED (SKIP_QUANT=1) ==="; else
    echo "=== PHASE A: SpinQuant(R1,R2,R4) + GPTQ W4A4 oneshot + save_compressed ==="
    python - <<PY
import os, time, torch
t_start=time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

MODEL=os.environ["MODEL"]; OUT=os.environ["OUT"]
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"])

print(f"[load] {MODEL} bf16 on CPU ...", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu",
                                           low_cpu_mem_usage=True, trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Qwen3-0.6B ties lm_head to embed_tokens. llmcompressor SpinQuant calls ct
# untie_word_embeddings, but under transformers 5.x that only flips the config flag --
# the tensors STAY shared, so centering+R1 hit one shared tensor and save either drops
# lm_head (silent garbage on reload) or raises on undeclared shared tensors.
# Manually untie with a real clone BEFORE oneshot.
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
recipe=[
    SpinQuantModifier(rotations=["R1","R2","R4"], transform_type="random-hadamard"),
    GPTQModifier(config_groups={"group_0": w4a4}, ignore=["lm_head"], actorder=None),
]
print(f"[quant] SpinQuant(R1,R2,R4 random-hadamard) + GPTQ W4A4 g128/token-dyn, N={N} seq={SEQ} ...", flush=True)
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
t_q=time.time(); print(f"[time] oneshot {t_q-t_data:.0f}s", flush=True)

# SpinQuant unties embeddings (Qwen3-0.6B is tied) and sets tie_word_embeddings=False,
# but transformers 5.x save_pretrained STILL drops _tied_weights_keys -> lm_head.weight
# lost from the checkpoint -> reload random-inits it = garbage logits. Clear the class
# tied-keys marker so the untied rotated lm_head actually serializes.
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
print("DONE_W4A4_SMOKE_QUANT", OUT, flush=True)
PY
    fi

    echo "=== saved config.json quantization_config + transform_config ==="
    python - <<PY
import json, os
cfg=json.load(open(os.environ["OUT"]+"/config.json"))
qc=cfg.get("quantization_config","<ABSENT>")
print("--- quantization_config (transform_config is NESTED inside it) ---")
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

# Diagnostic: did the loader attach the online R4 transform to down_proj?
def count_hooks():
    return sum(1 for name,m in model.named_modules()
               if "down_proj" in name and (m._forward_pre_hooks or m._forward_hooks))
print(f"[diag] down_proj hooks after plain load: {count_hooks()}", flush=True)

# transformers does NOT apply the saved transform_config (only llmcompressor oneshot
# and vLLM do). Re-apply the ONLINE-only entries (location input/output); the
# weight-location entries are already fused into the saved weights -- re-applying
# them would double-rotate.
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
print("DONE_W4A4_SMOKE_GEN", flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="; du -sh "$OUT" 2>/dev/null
