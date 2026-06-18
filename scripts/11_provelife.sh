#!/usr/bin/env bash
# Definitive B70 proof-of-life: generate with a KNOWN-GOOD standard model via
# llama-completion (not llama-bench, which crashes on this image). If this prints
# tokens, the B70 SYCL compute pipeline works and the Qwen3.6 crash is arch-specific.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/bartowski_Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
mkdir -p "$ROOT/.sycl_cache"

echo "############ 7B GPU generate via llama-completion (-ngl 99, NO flash-attn) ############"
docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 99 -c 4096 -fit off --no-warmup -n 64 -no-cnv --no-display-prompt \
  -p "List three uses for a GPU:" 2>&1 | tail -40
echo "=== exit ${PIPESTATUS[0]} ==="
