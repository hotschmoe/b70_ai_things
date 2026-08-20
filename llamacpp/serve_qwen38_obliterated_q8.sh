#!/usr/bin/env bash
# Pliny OBLITERATUS Qwen3.8-27B Q8_0 on 1x/2x B70 via 0xSero SYCL image.
# Card recipe (Pliny): temp 0, repeat_penalty 1.15, thinking off, no system
# prompt. n_predict stays unlimited (>=2048). Open WebUI workspace model
# qwen38-27b-obliterated-q8 pins the same params + max_tokens 8192.
# Q8 fused doors ON. Q4K doors OFF. vLLM P2P stays off.
#
#   ./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q8.sh start
#   GPU_COUNT=1 ./bin/gpu-run --card 0 bash llamacpp/serve_qwen38_obliterated_q8.sh start
#   bash llamacpp/serve_qwen38_obliterated_q8.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_oblit_q8}"
PORT="${PORT:-8010}"
GPU_COUNT="${GPU_COUNT:-2}"
CTX_SIZE_OVERRIDE="${CTX_SIZE_OVERRIDE:-32768}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
Q8_DOORS="${Q8_DOORS:-1}"
HOST_MODELS="${HOST_MODELS:-$REPO/models/files/qwen3.8-27b/obliterated-q8}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-OBLITERATED-Q8_0.gguf}"
OVERLAY="${OVERLAY:-$REPO/llamacpp/obliterated_q8_entrypoint.sh}"
IMG="${IMG:-qwen38-b70:latest}"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && say "stopped $NAME" || say "$NAME not running"
  exit 0
fi

[ -s "$HOST_MODELS/$MODEL_FILE" ] || { echo "missing $HOST_MODELS/$MODEL_FILE"; exit 2; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "missing image $IMG"; exit 2; }
[ -x "$OVERLAY" ] || chmod +x "$OVERLAY"

if [ "$GPU_COUNT" = "1" ]; then
  HEALTH_IMG="${HEALTH_IMG:-vllm-xpu-env:int8g-v0260}"
  "$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 90 2>&1 | tail -5 \
    || { say "UNHEALTHY card0"; exit 3; }
else
  "$REPO/bin/xpu-health" 2>&1 | tail -5 || { say "UNHEALTHY"; exit 3; }
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
say "P1 OBLITERATED Q8_0 GPU_COUNT=$GPU_COUNT Q8_DOORS=$Q8_DOORS ctx=$CTX_SIZE_OVERRIDE port=$PORT"
docker run -d --name "$NAME" \
  --device /dev/dri --ipc=host --shm-size 8g \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v "$HOST_MODELS:/models:ro" \
  -v "$OVERLAY:/entrypoint.sh:ro" \
  -e MODELS_DIR=/models \
  -e MODEL_FILE="$MODEL_FILE" \
  -e MODEL_SHA256="${MODEL_SHA256:-}" \
  -e GPU_COUNT="$GPU_COUNT" \
  -e CTX_SIZE_OVERRIDE="$CTX_SIZE_OVERRIDE" \
  -e PARALLEL=1 \
  -e BATCH="$BATCH" -e UBATCH="$UBATCH" \
  -e Q8_DOORS="$Q8_DOORS" \
  -e GGML_SYCL_FATTN_MMA=0 \
  -e GGML_SYCL_COMM_DIRECT_Q8="${GGML_SYCL_COMM_DIRECT_Q8:-2}" \
  -e GGML_SYCL_MMVQ_SG32="${GGML_SYCL_MMVQ_SG32:-0}" \
  -e GGML_SYCL_MMVQ_Q8_QUAD_SG24="${GGML_SYCL_MMVQ_Q8_QUAD_SG24:-1}" \
  -e LLAMA_P2P="${LLAMA_P2P:-0}" \
  -e THREADS=8 \
  -p "${PORT}:8010" \
  --entrypoint bash \
  "$IMG" /entrypoint.sh >/dev/null

say "waiting for :$PORT"
ok=0
for i in $(seq 1 180); do
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
     || curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ok=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    say "EXITED"
    docker logs "$NAME" 2>&1 | tail -80
    exit 1
  fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "UNHEALTHY after 15 min"
  docker logs "$NAME" 2>&1 | tail -80
  exit 1
fi
say "healthy $NAME on :$PORT served=$MODEL_FILE"
