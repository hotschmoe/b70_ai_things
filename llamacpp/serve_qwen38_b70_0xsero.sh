#!/usr/bin/env bash
# Campaign M2: 0xSero/qwen38-b70 llama.cpp SYCL Q4_K_M TP=2 @ native 262k.
# Thin wrapper around their compose tree (runtime clone, not a shelf).
#
#   ./bin/gpu-run bash llamacpp/serve_qwen38_b70_0xsero.sh start
#   bash llamacpp/serve_qwen38_b70_0xsero.sh stop
#
# LAB_DOORS=1 is the 2026-08-18 chase: lab Q4K reorder + SwiGLU fusion
# raised code c1 32.8 -> 43.8 at native 262k, Paris/fib still coherent.
# 0xSero's published entrypoint zeroes those doors (JIT quality guard).
# FATTN_MMA=1 crash-loops this JIT image; leave it 0.
# ENABLE_MTP=0: their hard-task MTP is a net loss.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
SRC="${SRC:-/mnt/vm_8tb/b70/qwen38-b70}"
NAME="${NAME:-qwen38-b70}"
PORT="${PORT:-8010}"
GPU_COUNT="${GPU_COUNT:-2}"
ENABLE_MTP="${ENABLE_MTP:-0}"
ENABLE_VISION="${ENABLE_VISION:-0}"
LAB_DOORS="${LAB_DOORS:-1}"
CTX_SIZE_OVERRIDE="${CTX_SIZE_OVERRIDE:-262144}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
OVERLAY="${OVERLAY:-$REPO/llamacpp/qwen38_b70_entrypoint_overlay.sh}"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ "$ACTION" = stop ]; then
  if [ -d "$SRC" ]; then
    (cd "$SRC" && docker compose down --remove-orphans) || true
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  say "stopped $NAME"
  exit 0
fi

[ -d "$SRC" ] || { echo "missing $SRC -- clone https://github.com/0xSero/qwen38-b70"; exit 2; }
[ -s "$SRC/models/Qwen3.8-27B-Q4_K_M.gguf" ] || { echo "missing GGUF at $SRC/models/"; exit 2; }
docker image inspect qwen38-b70:latest >/dev/null 2>&1 || { echo "missing image qwen38-b70:latest -- docker compose build in $SRC"; exit 2; }

say "pre-flight xpu-health"
"$REPO/bin/xpu-health" 2>&1 | tail -5 || { say "UNHEALTHY -- abort"; exit 3; }

ctx_show="${CTX_SIZE_OVERRIDE:-262144}"
say "M2 0xSero SYCL Q4_K_M GPU_COUNT=$GPU_COUNT ENABLE_MTP=$ENABLE_MTP LAB_DOORS=$LAB_DOORS ctx=$ctx_show batch=$BATCH/$UBATCH port=$PORT"
# compose hardcodes GPU_COUNT/CTX and the image entrypoint zeroes Q4K doors.
# docker run + overlay so campaign A/Bs can match the lab record flags.
docker rm -f "$NAME" >/dev/null 2>&1 || true
(cd "$SRC" && docker compose down --remove-orphans >/dev/null 2>&1) || true
docker run -d --name "$NAME" --restart unless-stopped \
  --device /dev/dri --ipc=host --shm-size 8g \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v "$SRC/models:/models" \
  -v "$OVERLAY:/entrypoint.sh:ro" \
  -e MODELS_DIR=/models \
  -e GPU_COUNT="$GPU_COUNT" \
  -e CTX_SIZE_OVERRIDE="$CTX_SIZE_OVERRIDE" \
  -e PARALLEL=1 \
  -e BATCH="$BATCH" -e UBATCH="$UBATCH" \
  -e ENABLE_MTP="$ENABLE_MTP" -e ENABLE_VISION="$ENABLE_VISION" \
  -e LAB_DOORS="$LAB_DOORS" \
  -e GGML_SYCL_FATTN_MMA="${GGML_SYCL_FATTN_MMA:-0}" \
  -e THREADS=8 \
  -p "${PORT}:8010" \
  qwen38-b70:latest >/dev/null

say "waiting for :$PORT /health or /v1/models"
ok=0
for i in $(seq 1 180); do
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
     || curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "UNHEALTHY after 15 min -- last 80 log lines:"
  docker logs --tail 80 "$NAME" 2>&1 || true
  exit 4
fi
say "UP on :$PORT"
curl -sS -m 5 "http://127.0.0.1:${PORT}/v1/models" || true
echo
