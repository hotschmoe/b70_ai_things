#!/usr/bin/env bash
# Prove the B70 SYCL inference pipeline with a known-good standard-attention model
# (Qwen2.5-7B-Instruct Q4_K_M). Downloads if missing, then runs llama-bench on GPU.
# This gives our FIRST real B70 tok/s number and isolates the Qwen3.6 arch crash.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
REPO="bartowski/Qwen2.5-7B-Instruct-GGUF"
FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"
DEST="$ROOT/models/${REPO//\//_}"
mkdir -p "$ROOT/.sycl_cache"

if [ ! -f "$DEST/$FILE" ]; then
  echo "=== downloading $REPO/$FILE ==="
  docker run --rm -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
    python:3.11 bash -c "pip install -q huggingface_hub && python - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='$REPO', filename='$FILE', local_dir='/models/${REPO//\//_}')
print('downloaded', p)
PY"
else
  echo "=== model already present ==="
fi

MODEL="/models/${REPO//\//_}/$FILE"
echo "=== llama-bench on B70 (SYCL) ==="
docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-bench "$IMG" \
  -m "$MODEL" -ngl 99 -p 512 -n 128 -fa 1 2>&1 | tee "$ROOT/results/knowngood_qwen25-7b_$(date +%Y%m%d_%H%M%S).txt"
echo "=== exit ${PIPESTATUS[0]} ==="
