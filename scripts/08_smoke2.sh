#!/usr/bin/env bash
# Clean smoke test: always pass --device, pin a SMALL context (avoid the auto-fit
# 245k-ctx KV OOM at warmup), skip warmup, no flash-attn. Want: tokens out + timings.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
mkdir -p "$ROOT/.sycl_cache"

docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 99 -c 4096 -fit off --no-warmup -n 48 -no-cnv --no-display-prompt \
  -p "Explain what a GPU is in exactly one sentence." 2>&1
echo "=== docker exit: $? ==="
