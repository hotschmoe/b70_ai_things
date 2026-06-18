#!/usr/bin/env bash
# Produce a compressed-tensors W4A8-INT checkpoint of Qwen3-14B that routes to the B70's
# XPUW4A8IntLinearKernel (oneDNN int4_gemm_w4a8) -- the ONLY upstream XPU kernel that lights the
# INT8 XMX datapath. Kernel reqs (verified in vllm source): int4 SYMMETRIC GROUP-quantized weights
# (group multiple of 32 -> 128), per-token DYNAMIC int8 SYMMETRIC activations, in/out dims mult of 8.
# vLLM detects this as CompressedTensorsW4A8Int (_is_dynamic_token_w4a8_int: w=4bit, a=8bit, dynamic).
# Data-free RTN (int4 group RTN + dynamic int8 act => no calibration). CPU, isolated python:3.11,
# all artifacts on /mnt/vm_8tb. ~2-3 min.
#   Env: OUTNAME (default Qwen3-14B-W4A8-INT), GROUP (default 128).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
SRC="${SRC:-/specula_models/Qwen3-14B}"
OUTNAME="${OUTNAME:-Qwen3-14B-W4A8-INT}"; GROUP="${GROUP:-128}"
LOG="$ROOT/results/quantize_w4a8_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"
echo "=== W4A8-INT quantize: src=$SRC out=$OUTNAME group=$GROUP  log=$LOG ==="

docker run --rm --name w4a8_quant \
  -v "$SPECULA:/specula_models:ro" -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/pip_cache:/root/.cache/pip" \
  -e HF_HOME=/hf_cache -e OMP_NUM_THREADS=32 \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
  -e SRC="$SRC" -e OUTNAME="$OUTNAME" -e GROUP="$GROUP" \
  python:3.11 bash -c '
    set -e
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q "llmcompressor>=0.8.0" "compressed-tensors" "transformers>=4.52" accelerate
    python - <<PY
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

SRC=os.environ["SRC"]; OUT="/models/"+os.environ["OUTNAME"]; G=int(os.environ["GROUP"])
print(f"[load] {SRC} (bf16, cpu)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True)
tok=AutoTokenizer.from_pretrained(SRC)

# Explicit W4A8-INT scheme: int4 sym group-G weights + int8 sym DYNAMIC per-token activations.
scheme=QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                             strategy="group", group_size=G),
    input_activations=QuantizationArgs(num_bits=8, type="int", symmetric=True,
                                       dynamic=True, strategy="token"),
)
recipe=[QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])]
print(f"[quant] DATA-FREE RTN W4A8-INT (int4 sym g{G} + dynamic int8 token act)...", flush=True)
oneshot(model=model, recipe=recipe)

print(f"[save] {OUT} (compressed)...", flush=True)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("DONE_W4A8", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit: ${PIPESTATUS[0]} ==="
du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
echo "--- quantization_config in config.json (verify scheme) ---"
docker run --rm -v "$ROOT/models:/models:ro" --entrypoint bash python:3.11 -c "grep -A30 quantization_config /models/$OUTNAME/config.json 2>/dev/null | head -35" || true
