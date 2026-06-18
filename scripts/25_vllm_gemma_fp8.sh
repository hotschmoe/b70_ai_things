#!/usr/bin/env bash
# Serve Gemma 4 12B with ONLINE FP8 quant on the single B70 (8-bit fast path demo).
# Standard attention -> no DeltaNet bugs. Fits easily (~13GB FP8) -> generous KV.
# All runtime caches on SSD. Online FP8 quant uses the well-tested path (not the
# broken serialized-fp8-checkpoint path).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="intel/llm-scaler-vllm:0.14.0-b8.3.1"
MODEL="/models/google_gemma-4-12B-it"
NAME="vllm_gemma_fp8"; PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}
docker rm -f "$NAME" 2>/dev/null || true

echo "=== launching $NAME (Gemma 4 12B, online FP8) ==="
docker run -d --name "$NAME" --device /dev/dri --shm-size 16g \
  -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" \
  -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e ZE_AFFINITY_MASK=0 \
  -e DISABLE_ESIMD_FUSED_INPUT=1 \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" \
    --served-model-name gemma4-fp8 \
    --host 0.0.0.0 --port ${PORT} \
    --quantization fp8 \
    --tensor-parallel-size 1 --dtype float16 --enforce-eager \
    --max-model-len 8192 --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code

echo "=== waiting for readiness (online FP8 quant + load, up to ~10 min) ==="
ok=0
for i in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  if docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited; then echo "container exited early"; break; fi
  sleep 5
done
echo "=== last 50 log lines ==="
docker logs "$NAME" 2>&1 | tail -50
if [ "$ok" = 1 ]; then
  echo "=== HEALTHY: quant method / kernel in logs ==="
  docker logs "$NAME" 2>&1 | grep -iE 'quant|fp8|int8|esimd|gemv|LinearMethod|XPU|XMX|woq' | grep -viE 'OperatorEntry|registered|dispatch' | tail -15
  echo "=== xpu-smi mem ==="; docker exec "$NAME" xpu-smi stats -d 0 2>/dev/null | grep -iE 'memory used|util' | head -6 || true
  echo "SERVER UP on :${PORT} as gemma4-fp8"
else
  echo "NOT HEALTHY — see logs above"
fi
