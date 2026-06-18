#!/usr/bin/env bash
# Isolate the segfault: build version, model arch metadata, CPU-only load,
# then GPU load without flash-attn.
ROOT=/mnt/vm_8tb/b70
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"
MODEL="/models/unsloth_Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
COMMON=(--rm -v "$ROOT/models:/models" -v "$ROOT/.sycl_cache:/sycl_cache")

echo "############ 1) build version ############"
docker run "${COMMON[@]}" --entrypoint /app/llama-cli "$IMG" --version 2>&1 | head -5

echo "############ 2) GGUF metadata (architecture, etc.) ############"
docker run "${COMMON[@]}" --entrypoint /app/llama-gguf "$IMG" "$MODEL" r n 2>&1 | grep -iE "architecture|general.name|block_count|context_length|expert|embedding_length|=" | head -25

echo "############ 3) CPU-ONLY load + 8 tokens (-ngl 0) ############"
docker run "${COMMON[@]}" --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 0 -n 8 -p "Hello" 2>&1 | tail -30
echo "exit(cpu)=$?"

echo "############ 4) GPU load, NO flash-attn, 8 tokens (-ngl 99) ############"
docker run "${COMMON[@]}" --device /dev/dri \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/sycl_cache \
  --entrypoint /app/llama-completion "$IMG" \
  -m "$MODEL" -ngl 99 -n 8 -p "Hello" 2>&1 | tail -40
echo "exit(gpu)=$?"
echo "############ DONE ############"
