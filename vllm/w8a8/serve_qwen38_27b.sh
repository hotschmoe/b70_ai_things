#!/usr/bin/env bash
# Research serve for on-box Qwen3.8-27B W8A8-gptq (scripts/150).
# NOT a shelf entry -- wraps the 3.6 W8A8 recipe with a 3.8 CKPT/SERVED.
# First gate: MTP off. Promote only after measured coherent + faster-or-equal.
#
#   B70_NOMTP=1 TP=2 PORT=18080 NAME=qwen38_w8a8 MAXLEN=262144 \
#     ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
#   bash vllm/w8a8/serve_qwen38_27b.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ACTION="${1:-start}"
if [ "$ACTION" = stop ]; then
  NAME="${NAME:-qwen38_w8a8}"
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

export CKPT="${CKPT:-/models/qwen3.8-27b/w8a8-gptq}"
export SERVED="${SERVED:-qwen3.8-27b-W8A8-gptq}"
export NAME="${NAME:-qwen38_w8a8}"
export PORT="${PORT:-18080}"
export TP="${TP:-2}"
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export B70_NOMTP="${B70_NOMTP:-1}"
export MAXLEN="${MAXLEN:-229376}"
export UTIL="${UTIL:-0.90}"
export GRAPH="${GRAPH:-1}"
export PREFIXCACHE="${PREFIXCACHE:-1}"
export KV_FP8="${KV_FP8:-0}"
# B1 (LOOP 6 leftover): KV_FP8=1 -> fp8_e5m2 storage KV. Default 0 = bf16
# (3.6 W8A8 serve.sh never mapped this env). Do not overwrite w8a8-gptq.
if [ "$KV_FP8" = 1 ]; then
  export KVDTYPE="${KVDTYPE:-fp8_e5m2}"
  echo "=== KV_FP8=1 -> --kv-cache-dtype $KVDTYPE ===" >&2
else
  export KVDTYPE="${KVDTYPE:-}"
fi

HOST_CKPT="$REPO/models/files/${CKPT#/models/}"
[ -d "$HOST_CKPT" ] || { echo "MISSING $HOST_CKPT"; exit 1; }

exec bash "$REPO/rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh" "$ACTION"
