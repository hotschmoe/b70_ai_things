#!/usr/bin/env bash
# Isolate SYCL-op crash: compute the SAME model on CPU (-ngl 0) while SYCL backend
# is still loaded (--device present so SYCL init succeeds). If CPU generates fine
# but -ngl 99 segfaults, the GPU SYCL kernels lack/mishandle this arch's ops
# (suspected Gated-DeltaNet). Only offload 4 layers to GPU in a 2nd run to bisect.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
mkdir -p "$ROOT/.sycl_cache"
run() {
  docker run --rm --device /dev/dri \
    -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
    -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
    --entrypoint /app/llama-completion "$IMG" \
    -m "$MODEL" -c 1024 -fit off --no-warmup -n 24 -no-cnv --no-display-prompt \
    -p "Explain what a GPU is in one sentence." "$@" 2>&1 | tail -20
}
echo "############ CPU compute (-ngl 0) ############"
run -ngl 0; echo "=== exit ${PIPESTATUS[0]} ==="
echo "############ GPU compute, 4 layers only (-ngl 4) ############"
run -ngl 4; echo "=== exit ${PIPESTATUS[0]} ==="
echo "############ DONE ############"
