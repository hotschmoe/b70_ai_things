#!/usr/bin/env bash
# Does Qwen3.6 Gated-DeltaNet compute on CPU at all in build 9680? Clean cache,
# -ngl 0. If CPU generates but GPU(-ngl 99) crashes => DeltaNet is a SYCL-only gap
# (build-from-source or Vulkan fixes it). If CPU also crashes => whole build lacks it.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
rm -rf "$ROOT/.sycl_cache"; mkdir -p "$ROOT/.sycl_cache"

echo "=== Qwen3.6-27B CPU compute (-ngl 0), clean cache ==="
docker run --rm --device /dev/dri \
  -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 0 -c 1024 -fit off --no-warmup -n 12 -no-cnv --no-display-prompt \
  -p "The capital of France is" 2>&1 | grep -vE 'repeat_last_n|dry_|top_k|mirostat|sampler (params|chain)' | tail -18
echo "=== exit ${PIPESTATUS[0]} ==="
