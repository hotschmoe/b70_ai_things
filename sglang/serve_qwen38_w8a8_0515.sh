#!/usr/bin/env bash
# P0.5: Qwen3.8-27B W8A8-gptq + NEXTN on sglang 0.5.15 (sglang-xpu:mtp-0515).
# Thin wrapper of serve_w8a8_0515.sh. Same 3.6 W8A8+NEXTN recipe; 3.8 is still
# Qwen3_5ForConditionalGeneration / qwen3_5 with grafted vision+MTP (num_nextn=1).
# Smoke = loads + Paris. DSpark on sglang-XPU is Phase 3, not this script.
# Do not overwrite models/files/qwen3.8-27b/w8a8-gptq.
#
#   PORT=18080 NAME=qwen38_w8a8_sglang MAXLEN=8192 \
#     ./bin/gpu-run bash sglang/serve_qwen38_w8a8_0515.sh start
#   bash sglang/serve_qwen38_w8a8_0515.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACTION="${1:-start}"
if [ "$ACTION" = stop ]; then
  NAME="${NAME:-qwen38_w8a8_sglang}"
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

export IMG="${IMG:-sglang-xpu:mtp-0515}"
export NAME="${NAME:-qwen38_w8a8_sglang}"
export CKPT="${CKPT:-/models/qwen3.8-27b/w8a8-gptq}"
export TOK="${TOK:-/models/qwen3.8-27b/w8a8-gptq}"
export SERVED="${SERVED:-qwen3.8-27b-W8A8-gptq-nextn}"
export PORT="${PORT:-18080}"
export MAXLEN="${MAXLEN:-8192}"
export SPEC_STEPS="${SPEC_STEPS:-10}"
export SPEC_DRAFT="${SPEC_DRAFT:-11}"

HOST_CKPT="$REPO/models/files/${CKPT#/models/}"
[ -d "$HOST_CKPT" ] || { echo "MISSING $HOST_CKPT"; exit 1; }

echo "=== P0.5 sglang 0.5.15 W8A8 3.8 NEXTN  steps=$SPEC_STEPS  ctx=$MAXLEN  name=$NAME ==="
echo "=== SERVED=$SERVED CKPT=$CKPT IMG=$IMG PORT=$PORT ==="
exec bash "$REPO/sglang/serve_w8a8_0515.sh" "$ACTION"
