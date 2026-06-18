#!/usr/bin/env bash
# Serve the self-made W8A8 INT8 Qwen3-14B checkpoint on vLLM-XPU (v0230) with DEBUG logging,
# then grep the startup log for the GROUND-TRUTH kernel/LinearMethod selection. This answers
# the empirical question (docs/literature/05_w8a8_recipe.md predicted source-level, never run):
#   does compressed-tensors W8A8 INT8 on Battlemage ...
#     (a) hit an XMX INT8 kernel,  (b) dequant-fall-back to FP16,  or  (c) hard-fail to find a kernel?
# Stops the live FP8 server first (single B70). Restore FP8 later with 36_serve.sh QUANT=fp8.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
CKPT="${CKPT:-/models/Qwen3-14B-W8A8-INT8}"      # path inside container (ROOT/models mounted at /models)
SERVED="${SERVED:-qwen3-14b-w8a8}"
NAME="${NAME:-vllm_w8a8}"; PORT=18080
MAXLEN="${MAXLEN:-8192}"; UTIL="${UTIL:-0.90}"
QFLAG="${QFLAG:-}"          # set to "--quantization compressed-tensors" to force-detect for debugging
mkdir -p "$ROOT"/{vllm_cache,tmp_ssd,results}

echo "=== stopping ALL live vllm servers on the GPU (free port 18080 + VRAM) ==="
docker rm -f "$NAME" vllm_qwen3 vllm_w4a8 vllm_w8a8 2>/dev/null || true

echo "=== serve W8A8: ckpt=$CKPT served=$SERVED maxlen=$MAXLEN util=$UTIL qflag='${QFLAG:-auto-detect}' ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  --entrypoint vllm "$IMG" \
  serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port ${PORT} \
    $QFLAG --dtype float16 --tensor-parallel-size 1 --enforce-eager \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
    --no-enable-prefix-caching --trust-remote-code

echo "=== waiting for readiness (up to ~12 min) ==="
ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done

echo; echo "===== GROUND-TRUTH kernel / quant selection (the answer) ====="
docker logs "$NAME" 2>&1 | grep -iE \
  'ScaledMMLinearKernel|LinearKernel|LinearMethod|kernel that can implement|CompressedTensors|w4a8|w8a8|int8|Int8|fp8|Fp8|dequant|Failed to find|No available memory|quantization|Selected|Using.*[Kk]ernel' \
  | grep -viE 'OperatorEntry|registered|dispatch|VLLM_' | tail -40

echo; echo "===== model footprint / KV / errors ====="
docker logs "$NAME" 2>&1 | grep -iE 'Resolved architecture|Model loading took|model weights take|Available KV cache|GPU KV cache size|Maximum concurrency|Application startup complete|error|Traceback|out of memory|RuntimeError|ValueError' \
  | grep -viE 'OperatorEntry|registered' | tail -20

echo
if [ "$ok" = 1 ]; then echo "HEALTHY :$PORT '$SERVED' -- W8A8 SERVES. Now bench vs FP8 (35_sweep_bench.sh)."
else echo "NOT HEALTHY -- W8A8 did NOT serve. Last 30 log lines:"; docker logs "$NAME" 2>&1 | tail -30; fi
