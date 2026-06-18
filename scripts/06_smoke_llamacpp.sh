#!/usr/bin/env bash
# Smoke test: load Qwen3.6-27B Q4_K_M on the B70 and generate a few tokens with
# FULL logging. Reveals: model-load time (PCIe reality), GPU buffer alloc sizes,
# tok/s, and any error llama-bench was hiding behind the tee/pipe.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="${MODEL:-/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf}"
mkdir -p "$ROOT/.sycl_cache"

echo "=== timing model load + 32-token generation ==="
time docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" \
  -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-cli \
  "$IMG" \
  -m "$MODEL" -ngl 99 -fa 1 -no-cnv \
  -p "Explain what a GPU is in one sentence." -n 32 2>&1

echo "=== exit: $? ==="
