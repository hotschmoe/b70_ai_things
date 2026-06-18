#!/usr/bin/env bash
# Clear the POISONED SYCL cache, then re-test the real target Qwen3.6-27B with the
# known-working minimal env (config A). If it generates, the earlier "DeltaNet crash"
# was actually the corrupt cache. Fresh persistent cache so the slow JIT is saved once.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"

echo "=== clearing poisoned SYCL cache ==="
rm -rf "$ROOT/.sycl_cache"
mkdir -p "$ROOT/.sycl_cache"

echo "=== Qwen3.6-27B generate, clean cache, minimal env ==="
docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 99 -c 4096 -fit off --no-warmup -n 48 -no-cnv --no-display-prompt \
  -p "List three uses for a GPU:" 2>&1 | grep -vE 'repeat_last_n|dry_|top_k|mirostat|sampler (params|chain)' | tail -25
echo "=== exit ${PIPESTATUS[0]} ==="
