#!/usr/bin/env bash
# Campaign M2: 0xSero/qwen38-b70 llama.cpp SYCL Q4_K_M TP=2 @ native 262k.
# Thin wrapper around their compose tree (runtime clone, not a shelf).
#
#   ./bin/gpu-run bash llamacpp/serve_qwen38_b70_0xsero.sh start
#   bash llamacpp/serve_qwen38_b70_0xsero.sh stop
#
# ENABLE_MTP=0 first: their own table says hard-task MTP is a net loss
# vs the 51 tok/s baseline. Flip ENABLE_MTP=1 only after the baseline
# gate (Paris + code + HE+).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
SRC="${SRC:-/mnt/vm_8tb/b70/qwen38-b70}"
NAME="${NAME:-qwen38-b70}"
PORT="${PORT:-8010}"
GPU_COUNT="${GPU_COUNT:-2}"
ENABLE_MTP="${ENABLE_MTP:-0}"
ENABLE_VISION="${ENABLE_VISION:-0}"

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

say "M2 0xSero SYCL Q4_K_M GPU_COUNT=$GPU_COUNT ENABLE_MTP=$ENABLE_MTP ctx=262144 port=$PORT"
(cd "$SRC" && GPU_COUNT="$GPU_COUNT" ENABLE_MTP="$ENABLE_MTP" ENABLE_VISION="$ENABLE_VISION" \
  docker compose up -d --no-build)

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
