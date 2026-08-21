#!/usr/bin/env bash
# Qwen3.8-27B W4A8 research serve (NOT a shelf). Clone of
# rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh via exec, 3.8 env overrides.
# Campaign: docs/20260820_qwen38_w4a8_campaign.md. GRAPH=0 first, Paris+fib.
# Served id must encode method+scheme. P2PACCESS=0. DD stays parked.
#
#   GRAPH=0 NOMM=1 B70_NOMTP=1 PORT=18081 NAME=qwen38_w4a8_rtn \
#     ./bin/gpu-run --card 1 bash vllm/w4a8/serve_qwen38_w4a8.sh start
#   bash vllm/w4a8/serve_qwen38_w4a8.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh"
ACTION="${1:-start}"

if [ "$ACTION" = stop ]; then
  NAME="${NAME:-qwen38_w4a8_rtn}"
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$SHELF" ] || { echo "missing 3.6 shelf serve $SHELF"; exit 1; }

export CKPT="${CKPT:-/models/qwen3.8-27b/w4a8-rtn-gdn}"
export SERVED="${SERVED:-qwen3.8-27b-W4A8-rtn-gdn}"
export NAME="${NAME:-qwen38_w4a8_rtn}"
export PORT="${PORT:-18081}"
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export TP="${TP:-1}"
export DEVICE="${DEVICE:-${CARD:-1}}"
export GRAPH="${GRAPH:-0}"
export DTYPE="${DTYPE:-float16}"
export NOMM="${NOMM:-1}"
export P2PACCESS="${P2PACCESS:-0}"
export UTIL="${UTIL:-0.85}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-2}"
export B70_W4A8_HYBRID="${B70_W4A8_HYBRID:-0}"
# First smoke: no MTP. Path H hybrid is a later A/B; HYBRID=0 proves int4_gemm_w4a8.
export B70_NOMTP="${B70_NOMTP:-1}"
export MTPTOK="${MTPTOK:-}"

echo "=== 3.8 W4A8 research serve SERVED=$SERVED GRAPH=$GRAPH NOMM=$NOMM DEVICE=$DEVICE PORT=$PORT ==="
echo "=== IMG=$IMG CKPT=$CKPT P2PACCESS=$P2PACCESS HYBRID=$B70_W4A8_HYBRID NOMTP=$B70_NOMTP ==="
exec bash "$SHELF" "$ACTION"
