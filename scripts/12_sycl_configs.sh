#!/usr/bin/env bash
# Find a config where the B70 actually executes a forward pass. The 7B crashes at
# first eval regardless of model -> SYCL compute config issue, not the model.
# Try: minimal env, OpenCL backend, Level-Zero immediate-cmdlists OFF, SYSMAN on.
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/bartowski_Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
mkdir -p "$ROOT/.sycl_cache"

try() {
  local name="$1"; shift
  echo "######################## CONFIG: $name ########################"
  docker run --rm --device /dev/dri \
    -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache" \
    "$@" \
    --entrypoint /app/llama-completion "$IMG" \
    -m "$MODEL" -ngl 99 -c 2048 -fit off --no-warmup -n 16 -no-cnv --no-display-prompt \
    -p "The capital of France is" 2>&1 | grep -vE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ I (sampler|system_info|generate|llama_completion: llama)|repeat_last_n|dry_|top_k|mirostat|sampler chain' | tail -12
  echo "   --> exit ${PIPESTATUS[0]}"
  echo
}

try "A: minimal env (level_zero default)" -e ONEAPI_DEVICE_SELECTOR=level_zero:0
try "B: OpenCL backend"                   -e ONEAPI_DEVICE_SELECTOR=opencl:gpu
try "C: L0 immediate-cmdlists OFF"        -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
try "D: SYSMAN + relaxed alloc"           -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZES_ENABLE_SYSMAN=1 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
echo "######################## DONE ########################"
