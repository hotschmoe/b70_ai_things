#!/usr/bin/env bash
# Produce a compressed-tensors W8A8 INT8 checkpoint of Qwen3-14B (dense, GQA) from the
# local BF16 weights, on CPU (XPU calibration is unreliable; INT8 kernels assume non-XPU).
# Scheme W8A8 = per-channel symmetric int8 weights + per-token DYNAMIC int8 activations.
# Recipe per docs/literature/05_w8a8_recipe.md. Runs in an isolated python:3.11 container
# (CPU torch) so it can't touch the GPU and won't clobber the XPU torch stack; safe to run
# alongside the live FP8 server. Long-running -> launch with nohup/background.
#
# Env knobs:
#   DATAFREE=1 (default) -> pure RTN W8A8 + per-token DYNAMIC int8 activations, NO SmoothQuant,
#       NO calibration data. Data-free, ~2-3 min on CPU. Produces a valid W8A8 artifact for the
#       kernel-path test. Slightly lower accuracy (no outlier smoothing) -- fine for verification.
#   DATAFREE=0 -> SmoothQuant + (RTN|GPTQ) calibrated. Higher accuracy but runs HOURS on CPU
#       (SmoothQuant does sequential per-block forward passes over SAMPLES). Use overnight only.
#   METHOD=rtn|gptq (only used when DATAFREE=0), SAMPLES, SEQLEN, SMOOTH.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
SRC="${SRC:-/specula_models/Qwen3-14B}"
DATAFREE="${DATAFREE:-1}"
METHOD="${METHOD:-rtn}"; SAMPLES="${SAMPLES:-256}"; SEQLEN="${SEQLEN:-2048}"; SMOOTH="${SMOOTH:-0.8}"
# Tag the output dir by method so RTN and GPTQ NEVER collide / get mixed up downstream (rtn vs gptq).
QMETH="$METHOD"; [ "$DATAFREE" = 1 ] && QMETH="rtn"
OUTNAME="${OUTNAME:-Qwen3-14B-W8A8-${QMETH}}"
LOG="$ROOT/results/quantize_w8a8_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"

echo "=== W8A8 quantize: src=$SRC out=$OUTNAME method=$METHOD samples=$SAMPLES seqlen=$SEQLEN ==="
echo "=== log: $LOG ==="

docker run --rm --name w8a8_quant \
  -v "$SPECULA:/specula_models:ro" -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/pip_cache:/root/.cache/pip" \
  -e HF_HOME=/hf_cache -e OMP_NUM_THREADS=32 \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
  -e SRC="$SRC" -e OUTNAME="$OUTNAME" -e METHOD="$METHOD" -e DATAFREE="$DATAFREE" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" \
  python:3.11 bash -c '
    set -e
    echo "[pip] installing CPU torch + llmcompressor (one-time, cached)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q "llmcompressor>=0.8.0" "compressed-tensors" "transformers>=4.52" datasets accelerate
    python - <<PY
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SRC=os.environ["SRC"]; OUT="/models/"+os.environ["OUTNAME"]
METHOD=os.environ["METHOD"].lower(); DATAFREE=os.environ.get("DATAFREE","1")=="1"
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])
print(f"[load] {SRC} (bf16, cpu)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True)
tok=AutoTokenizer.from_pretrained(SRC)

if DATAFREE:
    # Pure RTN weights + per-token DYNAMIC int8 activations => no calibration data, no SmoothQuant.
    print("[quant] DATA-FREE RTN W8A8 (int8 weights per-channel + dynamic per-token int8 act)...", flush=True)
    recipe=[QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
    oneshot(model=model, recipe=recipe)
else:
    from llmcompressor.modifiers.transform import SmoothQuantModifier
    print(f"[data] ultrachat_200k [:{N}]...", flush=True)
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
    def pp(e): return {"text": tok.apply_chat_template(e["messages"], tokenize=False)}
    ds=ds.map(pp)
    def tk(s): return tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False)
    ds=ds.map(tk, remove_columns=ds.column_names)
    if METHOD=="gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        q=GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
    else:
        q=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
    recipe=[SmoothQuantModifier(smoothing_strength=SMOOTH), q]
    print(f"[quant] CALIBRATED method={METHOD} scheme=W8A8 (SmoothQuant+{METHOD})...", flush=True)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)

print(f"[save] {OUT} (compressed)...", flush=True)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("DONE_W8A8", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"

echo "=== exit: ${PIPESTATUS[0]} ==="
du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
ls -la "$ROOT/models/$OUTNAME" 2>/dev/null | head -20
