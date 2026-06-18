#!/usr/bin/env bash
# Serve Gemma 4 12B on the single B70 via the from-source upstream vllm-xpu-env image.
# QUANT env selects path: fp8 (default, 8-bit fast path) | none (BF16 baseline) | sym_int4.
# Intel Gemma-4 recipe: --attention-backend TRITON_ATTN --enforce-eager. Multimodal -> mm flags.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${VLLM_IMG:-vllm-xpu-env:tf}"
MODEL="/models/google_gemma-4-12B-it"
NAME="vllm_gemma4"; PORT=18080
QUANT="${QUANT:-fp8}"   # fp8 | none | sym_int4
ATTN="${ATTN:-}"        # empty = vLLM default (SYCL-TLA flash); or TRITON_ATTN / FLASH_ATTN
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f "$NAME" 2>/dev/null || true

QARG="--quantization $QUANT"; [ "$QUANT" = "none" ] && QARG=""
ABARG=""; [ -n "$ATTN" ] && ABARG="--attention-backend $ATTN"

echo "=== launching $NAME (Gemma 4 12B, quant=$QUANT) on $IMG ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" \
  -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" \
    --served-model-name gemma4 \
    --host 0.0.0.0 --port ${PORT} \
    $QARG $ABARG \
    --tensor-parallel-size 1 --enforce-eager \
    --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0}' --no-enable-prefix-caching --trust-remote-code

echo "=== waiting for readiness (build-from-source first run may JIT; up to ~12 min) ==="
ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "exited early"; break; }
  sleep 5
done
echo "=== last 55 log lines ==="
docker logs "$NAME" 2>&1 | tail -55
if [ "$ok" = 1 ]; then
  echo "=== HEALTHY: quant/kernel evidence ==="
  docker logs "$NAME" 2>&1 | grep -iE 'quant|fp8|int8|esimd|gemv|LinearMethod|attention backend|triton|XMX|woq|Maximum concurrency|KV cache' | grep -viE 'OperatorEntry|registered|dispatch' | tail -18
  docker exec "$NAME" xpu-smi stats -d 0 2>/dev/null | grep -iE 'memory used|util' | head -6 || true
  echo "SERVER UP :${PORT} model 'gemma4' quant=$QUANT"
else echo "NOT HEALTHY — see logs"; fi
