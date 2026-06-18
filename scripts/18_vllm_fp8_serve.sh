#!/usr/bin/env bash
# Launch Qwen3.6-27B-FP8 on the single B70 via llm-scaler vLLM. ALL runtime caches/tmp
# redirected to the SSD so the near-full 50GB docker.img is NOT touched at runtime.
# Detached server + readiness wait. Benchmark in the next script once healthy.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${VLLM_IMG:-intel/llm-scaler-vllm:0.14.0-b8.3.1}"
MODEL="/models/Qwen_Qwen3.6-27B-FP8"
NAME="vllm_fp8"
PORT=18080
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd}

echo "=== cleaning any prior $NAME ==="
docker rm -f "$NAME" 2>/dev/null || true

echo "=== launching $NAME (detached) ==="
docker run -d --name "$NAME" --device /dev/dri --shm-size 16g \
  -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" \
  -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache \
  -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e ZE_AFFINITY_MASK=0 \
  -e DISABLE_ESIMD_FUSED_INPUT=1 \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" \
    --served-model-name qwen36-fp8 \
    --host 0.0.0.0 --port ${PORT} \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --enforce-eager \
    --block-size 64 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.97 \
    --no-enable-prefix-caching \
    --trust-remote-code

echo "=== waiting for readiness (model load + JIT, up to ~8 min) ==="
ok=0
for i in $(seq 1 96); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  # surface fatal errors early
  if docker logs "$NAME" 2>&1 | grep -qiE 'error|traceback|out of memory|failed|exception' ; then
    echo "--- possible error in logs at ${i}x5s ---"
  fi
  sleep 5
done

echo "=== last 40 log lines ==="
docker logs "$NAME" 2>&1 | tail -40
if [ "$ok" = 1 ]; then
  echo "=== HEALTHY. GPU mem snapshot (xpu-smi) ==="
  docker exec "$NAME" xpu-smi stats -d 0 2>/dev/null | grep -iE 'memory used|utilization|power' | head -10 || echo "(xpu-smi stats unavailable)"
  echo "SERVER UP on port ${PORT} as model 'qwen36-fp8'"
else
  echo "SERVER NOT HEALTHY after wait — inspect logs above."
fi
