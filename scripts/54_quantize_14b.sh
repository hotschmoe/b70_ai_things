#!/usr/bin/env bash
# Generalized Qwen3-14B quantizer (standard dense GQA) -> compressed-tensors, GPU-CAPABLE.
# Creates the quant formats we are MISSING for the eval matrix (W4A16, W8A16) and can also reproduce
# W8A8 / W4A8. 14B is a plain dense Qwen3 (NOT the hybrid Qwen3_5), so SmoothQuant's mapping resolver
# works here -- unlike the 27B (see scripts/49 + JOURNAL 2026-06-19). Mirrors scripts/49's GPU path:
# model stays on CPU (bf16 in 125 GB RAM), llmcompressor onloads one layer at a time to the B70 for the
# calibration forward passes. Run with the GPU EXCLUSIVE (no server up) -- VRAM contention device-losts.
#
# Env:
#   SCHEME   W4A16 (default) | W8A16 | W8A8 | W4A8  (llmcompressor preset scheme strings)
#   SRC      bf16 source (default /specula_models/Qwen3-14B)
#   OUTNAME  default Qwen3-14B-<SCHEME>
#   DEVICE   xpu (default) | cpu
#   METHOD   gptq (default) | rtn
#   SMOOTHQUANT  default 0 for *A16 weight-only (SmoothQuant smooths ACTIVATION outliers -> only helps
#                when activations are quantized; for A16 it is a no-op-ish cost). Set 1 for W8A8/W4A8.
#   SAMPLES (256) SEQLEN (2048) SMOOTH (0.8) IGNORE (lm_head)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:-/specula_models/Qwen3-14B}"
SCHEME="${SCHEME:-W4A16}"
DEVICE="${DEVICE:-xpu}"; METHOD="${METHOD:-gptq}"
SAMPLES="${SAMPLES:-256}"; SEQLEN="${SEQLEN:-2048}"; SMOOTH="${SMOOTH:-0.8}"
IGNORE="${IGNORE:-lm_head}"
# A16 (weight-only) schemes: SmoothQuant off by default; A8 schemes: on.
case "$SCHEME" in *A16) SQ_DEFAULT=0;; *) SQ_DEFAULT=1;; esac
SMOOTHQUANT="${SMOOTHQUANT:-$SQ_DEFAULT}"
DATAFREE="${DATAFREE:-0}"   # 1 = fast RTN (no calibration data/GPTQ) -> quick serveability test
# Tag the output dir by method so RTN and GPTQ NEVER collide / get mixed up downstream (rtn vs gptq).
QMETH="$METHOD"; [ "$DATAFREE" = 1 ] && QMETH="rtn"
OUTNAME="${OUTNAME:-Qwen3-14B-${SCHEME}-${QMETH}}"
LOG="$ROOT/results/quant14b_${SCHEME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"

GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="")
[ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0)
echo "=== 14B quant: scheme=$SCHEME device=$DEVICE method=$METHOD smoothquant=$SMOOTHQUANT samples=$SAMPLES out=$OUTNAME ==="
echo "=== log: $LOG ==="

docker run --rm --name quant14b "${GPUARGS[@]}" --ipc=host --shm-size 32g \
  -v "$ROOT:$ROOT" -v "$SPECULA:/specula_models:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME="$ROOT/vllm_cache" -e OMP_NUM_THREADS=32 -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e OUT="$ROOT/models/$OUTNAME" -e DEVICE="$DEVICE" -e METHOD="$METHOD" -e SCHEME="$SCHEME" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" -e IGNORE="$IGNORE" -e SMOOTHQUANT="$SMOOTHQUANT" -e DATAFREE="$DATAFREE" \
  --entrypoint bash "$IMG" -c '
    set -e
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -2 || true
    python - <<PY
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
try: from llmcompressor.modifiers.transform import SmoothQuantModifier
except Exception: from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

SRC=os.environ["SRC"]; OUT=os.environ["OUT"]; DEV=os.environ["DEVICE"]; METHOD=os.environ["METHOD"].lower()
SCHEME=os.environ["SCHEME"]; N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])
USE_SMOOTH=os.environ.get("SMOOTHQUANT","0")=="1"; DATAFREE=os.environ.get("DATAFREE","0")=="1"
IGN=[p for p in os.environ["IGNORE"].split() if p]
xpu_ok=hasattr(torch,"xpu") and torch.xpu.is_available()
print(f"[probe] xpu={xpu_ok} device={DEV} scheme={SCHEME} smoothquant={USE_SMOOTH} datafree={DATAFREE}", flush=True)
print(f"[load] {SRC} bf16 on CPU (llmcompressor onloads layers to the GPU per-step)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

t0=time.time()
if DATAFREE:
    # Fast RTN (round-to-nearest), no calibration data/GPTQ -> minutes. Good for a quick serveability test.
    recipe=[QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGN)]
    print(f"[quant] DATA-FREE RTN {SCHEME}, ignore={IGN} ...", flush=True)
    oneshot(model=model, recipe=recipe)
else:
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
    ds=ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
    ds=ds.map(lambda s: tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False),
              remove_columns=ds.column_names)
    if METHOD=="gptq":
        q=GPTQModifier(targets="Linear", scheme=SCHEME, ignore=IGN, actorder=None)  # actorder None: avoid XPU gather device-lost
    else:
        q=QuantizationModifier(targets="Linear", scheme=SCHEME, ignore=IGN)
    # A16 schemes are weight-only -> SmoothQuant brings little; default off (see header).
    recipe=[SmoothQuantModifier(smoothing_strength=SMOOTH), q] if USE_SMOOTH else [q]
    pfx = "SmoothQuant+" if USE_SMOOTH else ""
    print(f"[quant] {pfx}{METHOD} {SCHEME}, ignore={IGN} ...", flush=True)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
print(f"[done] quant {time.time()-t0:.0f}s; saving -> {OUT}", flush=True)
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
print("DONE_14B_QUANT", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="; du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
