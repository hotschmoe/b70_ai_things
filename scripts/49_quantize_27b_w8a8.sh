#!/usr/bin/env bash
# Quantize Qwen3.6-27B (DeltaNet) -> compressed-tensors W8A8 INT8 with GOOD calibration (SmoothQuant +
# GPTQ) to stay close to W8A16/BF16 accuracy. GPU-ACCELERATED: runs in the XPU image so llmcompressor
# uses the B70 ("Accelerator 0") for the per-layer calibration forward passes (the model itself stays on
# CPU -- 54GB BF16 in 125GB RAM -- and llmcompressor's SequentialPipeline onloads one layer at a time to
# the GPU). GPTQ actorder=False to avoid the H[perm][:,perm] gather that device-lost before.
#
# IGNORE list keeps the DeltaNet linear-attn projections + MTP head + lm_head in BF16 (quantizing them
# hurts accuracy / they are not group-divisible) -- only the standard attn/MLP linears go int8 (our kernel).
#
# Env: SRC (27B BF16 path), OUTNAME, DEVICE=xpu|cpu (default xpu), METHOD=gptq|rtn (default gptq),
#      SAMPLES (default 512), SEQLEN (default 2048), SMOOTH (0.8), IGNORE (space-sep patterns), DATAFREE=0.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:-/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B}"
OUTNAME="${OUTNAME:-Qwen3.6-27B-W8A8-INT8}"
DEVICE="${DEVICE:-xpu}"; METHOD="${METHOD:-gptq}"; SAMPLES="${SAMPLES:-512}"; SEQLEN="${SEQLEN:-2048}"; SMOOTH="${SMOOTH:-0.8}"
# DeltaNet/MTP/head stay BF16. Refine after inspecting the model's module names.
IGNORE="${IGNORE:-lm_head re:.*linear_attn.* re:.*\.mtp.* re:.*mtp\..*}"
LOG="$ROOT/results/quant27b_w8a8_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"
[ -d "$SRC" ] || { echo "MISSING 27B weights at $SRC (download Qwen/Qwen3.6-27B first)"; exit 1; }

GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="")
[ "$DEVICE" = xpu ] && GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0)
echo "=== 27B W8A8 quant: device=$DEVICE method=$METHOD samples=$SAMPLES ignore=[$IGNORE]  log=$LOG ==="

docker run --rm --name quant27b "${GPUARGS[@]}" --ipc=host --shm-size 32g \
  -v "$ROOT:$ROOT" -e HF_HOME=/hf_cache -e XDG_CACHE_HOME="$ROOT/vllm_cache" \
  -e OMP_NUM_THREADS=32 -e PIP_CACHE_DIR="$ROOT/pip_cache" \
  -e SRC="$SRC" -e OUT="$ROOT/models/$OUTNAME" -e DEVICE="$DEVICE" -e METHOD="$METHOD" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" -e IGNORE="$IGNORE" \
  --entrypoint bash "$IMG" -c '
    set -e
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -2 || true
    python - <<PY
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
try: from llmcompressor.modifiers.transform import SmoothQuantModifier
except Exception: from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

SRC=os.environ["SRC"]; OUT=os.environ["OUT"]; DEV=os.environ["DEVICE"]; METHOD=os.environ["METHOD"].lower()
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])
IGN=[p for p in os.environ["IGNORE"].split() if p]
xpu_ok=hasattr(torch,"xpu") and torch.xpu.is_available()
print(f"[probe] xpu={xpu_ok} device={DEV}", flush=True)
print(f"[load] {SRC} bf16 on CPU (llmcompressor onloads layers to the GPU per-step)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
ds=ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
ds=ds.map(lambda s: tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False),
          remove_columns=ds.column_names)
if METHOD=="gptq":
    q=GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGN, actorder=False)  # actorder off: avoids XPU gather device-lost
else:
    q=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=IGN)
recipe=[SmoothQuantModifier(smoothing_strength=SMOOTH), q]
print(f"[quant] SmoothQuant+{METHOD} W8A8, ignore={IGN} ...", flush=True)
t0=time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
print(f"[done] calib+quant {time.time()-t0:.0f}s; saving compressed -> {OUT}", flush=True)
model.save_pretrained(OUT, save_compressed=True); tok.save_pretrained(OUT)
print("DONE_27B_W8A8", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="; du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
