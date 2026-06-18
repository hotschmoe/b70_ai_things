#!/usr/bin/env bash
# EXPERIMENT: can the B70 accelerate calibration-based quantization (SmoothQuant/GPTQ forward
# passes) vs CPU? Runs llmcompressor inside the XPU image (torch-xpu) with the model placed on
# the B70, on a SMALL calib config, and times it. Run twice (DEVICE=cpu then DEVICE=xpu) to compare.
# NOTE: data-free RTN has no forward passes -> GPU can't help it; this is for the CALIBRATED path.
# The B70 must be free -> stops the FP8 server first. All caches on /mnt/vm_8tb (SSD).
#
# Env: DEVICE=xpu|cpu (default xpu), SAMPLES (default 64), SEQLEN (default 512),
#      METHOD=gptq|rtn (default gptq -- the expensive path worth accelerating), SMOOTH (0.8).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:-/specula_models/Qwen3-14B}"
DEVICE="${DEVICE:-xpu}"; SAMPLES="${SAMPLES:-64}"; SEQLEN="${SEQLEN:-512}"
METHOD="${METHOD:-gptq}"; SMOOTH="${SMOOTH:-0.8}"
OUTNAME="${OUTNAME:-Qwen3-14B-W8A8-INT8-${DEVICE}probe}"
LOG="$ROOT/results/quant_on_${DEVICE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models"

if [ "$DEVICE" = xpu ]; then
  echo "=== freeing the B70 (stopping ALL vllm servers -> exclusive GPU) for the XPU quant experiment ==="
  docker rm -f vllm_qwen3 vllm_w4a8 vllm_w8a8 2>/dev/null || true
  GPUARGS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0)
else
  GPUARGS=(-e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="")
fi

echo "=== quant-on-$DEVICE: method=$METHOD samples=$SAMPLES seqlen=$SEQLEN img=$IMG ==="
echo "=== log: $LOG ==="
docker run --rm --name quant_probe "${GPUARGS[@]}" --ipc=host --shm-size 16g \
  -v "$SPECULA:/specula_models:ro" -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/vllm_cache -e OMP_NUM_THREADS=32 \
  -e SRC="$SRC" -e OUTNAME="$OUTNAME" -e DEVICE="$DEVICE" -e METHOD="$METHOD" \
  -e SAMPLES="$SAMPLES" -e SEQLEN="$SEQLEN" -e SMOOTH="$SMOOTH" \
  --entrypoint bash "$IMG" -c '
    set -e
    echo "[pip] adding llmcompressor on top of the image torch-xpu (deps ok; ephemeral container)..."
    pip install -q "llmcompressor>=0.8.0" datasets accelerate 2>&1 | tail -3 || true
    python - <<PY
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
try:
    from llmcompressor.modifiers.transform import SmoothQuantModifier
except Exception:
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

DEV=os.environ["DEVICE"]; SRC=os.environ["SRC"]; OUT="/models/"+os.environ["OUTNAME"]
METHOD=os.environ["METHOD"].lower()
N=int(os.environ["SAMPLES"]); SEQ=int(os.environ["SEQLEN"]); SMOOTH=float(os.environ["SMOOTH"])

# ---- PROBE: is the B70 visible to torch as an accelerator? ----
xpu_ok = hasattr(torch,"xpu") and torch.xpu.is_available()
print(f"[probe] torch={torch.__version__} torch.xpu.is_available()={xpu_ok}", flush=True)
if xpu_ok:
    print(f"[probe] xpu device 0 = {torch.xpu.get_device_name(0)}", flush=True)
try:
    from llmcompressor.pytorch.utils.helpers import get_main_device  # path varies by version
    print(f"[probe] llmcompressor get_main_device() = {get_main_device()}", flush=True)
except Exception as e:
    print(f"[probe] get_main_device probe skipped: {e}", flush=True)

want_xpu = (DEV=="xpu" and xpu_ok)
dmap = "xpu" if want_xpu else "cpu"
ld = "float16" if want_xpu else "bfloat16"
print(f"[load] {SRC} dtype={ld} device_map={dmap}...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC,
        torch_dtype=(torch.float16 if want_xpu else torch.bfloat16),
        device_map=dmap, low_cpu_mem_usage=True)
tok=AutoTokenizer.from_pretrained(SRC)

ds=load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
def pp(e): return {"text": tok.apply_chat_template(e["messages"], tokenize=False)}
ds=ds.map(pp)
def tk(s): return tok(s["text"], padding=False, max_length=SEQ, truncation=True, add_special_tokens=False)
ds=ds.map(tk, remove_columns=ds.column_names)

if METHOD=="gptq":
    q=GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
else:
    q=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
recipe=[SmoothQuantModifier(smoothing_strength=SMOOTH), q]

print(f"[quant] device={dmap} method={METHOD} N={N} seq={SEQ} -- timing...", flush=True)
t0=time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N)
dt=time.time()-t0
print(f"[RESULT] calib+quant wall-clock on {dmap}: {dt:.1f}s  ({N} samples x {SEQ} tok, {METHOD})", flush=True)
print("DONE_QUANT_PROBE", dmap, f"{dt:.1f}s", flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit ${PIPESTATUS[0]} ==="
grep -E "\[probe\]|\[RESULT\]|DONE_QUANT_PROBE|Error|Traceback" "$LOG" | tail -15
