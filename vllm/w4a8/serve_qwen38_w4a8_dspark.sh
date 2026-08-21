#!/usr/bin/env bash
# Qwen3.8-27B W4A8-gptq-gdn + off-shelf DSpark (NOT a shelf).
# Campaign K17: docs/20260820_qwen38_w4a8_campaign.md section 11.
# Clone of vllm/dflash/serve_qwen38_w8a8_dspark.sh onto the 3.8 W4A8
# research serve. Readout fix + Qwen3DSparkModel remap from
# vllm/dflash/patches/v0260. method=dspark (not dflash).
# GRAPH=0 first (W8A8 GRAPH=1 + spec died mid-HE+). KV auto/bf16 (D13).
# P2PACCESS=0. Do not stop the TP=1 GRAPH=1 score serve on the other card.
#
#   GRAPH=0 SPECTOK=7 PORT=18083 NAME=qwen38_w4a8_dspark DEVICE=1 \
#     ./bin/gpu-run --card 1 bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
#   NAME=qwen38_w4a8_dspark bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_w4a8_dspark}"
PATCH="$REPO/vllm/dflash/patches/v0260"

if [ "$ACTION" = stop ]; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$PATCH/dflash.py" ] || { echo "missing $PATCH/dflash.py"; exit 1; }
[ -f "$PATCH/drafter_config.json" ] || { echo "missing $PATCH/drafter_config.json"; exit 1; }

SPECTOK="${SPECTOK:-7}"
DRAFTER_REL="${DRAFTER_REL:-qwen3.8-27b/dflash-drafter-fp8-b70}"
HOST_DRAFT="$REPO/models/files/${DRAFTER_REL}"
[ -d "$HOST_DRAFT" ] || { echo "MISSING $HOST_DRAFT"; exit 1; }

export CKPT="${CKPT:-/models/qwen3.8-27b/w4a8-gptq-gdn}"
export SERVED="${SERVED:-qwen3.8-27b-W4A8-gptq-dspark${SPECTOK}}"
export NAME
export PORT="${PORT:-18083}"
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export TP="${TP:-1}"
export DEVICE="${DEVICE:-${CARD:-1}}"
export GRAPH="${GRAPH:-0}"
export DTYPE="${DTYPE:-float16}"
export NOMM="${NOMM:-1}"
export P2PACCESS="${P2PACCESS:-0}"
export UTIL="${UTIL:-0.88}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-1}"
export B70_W4A8_HYBRID="${B70_W4A8_HYBRID:-0}"
# Do NOT set B70_NOMTP=1 -- that clears SPEC in the 3.6 W4A8 shelf.
export B70_NOMTP=0
export MTPTOK=""
export THINK_BUDGET="${THINK_BUDGET:-0}"
export SPEC="${SPEC:-{\"method\":\"dspark\",\"model\":\"/models/${DRAFTER_REL}\",\"num_speculative_tokens\":${SPECTOK}}}"
export B70_EXTRA_MOUNTS="${B70_EXTRA_MOUNTS:+$B70_EXTRA_MOUNTS }${PATCH}/dflash.py:/workspace/vllm/vllm/v1/spec_decode/dflash.py:ro ${PATCH}/utils.py:/workspace/vllm/vllm/v1/spec_decode/utils.py:ro ${PATCH}/drafter_config.json:/models/${DRAFTER_REL}/config.json:ro ${PATCH}/compile_key_sitecustomize.py:/opt/compile_key_shim/sitecustomize.py:ro ${PATCH}/compile_key_spectok_so.py:/opt/compile_key_shim/compile_key_spectok_so.py:ro"
export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }PYTHONPATH=/opt/compile_key_shim:/opt/push_ar:/opt/mtp_shim"

echo "=== K17 W4A8-gptq + DSpark k=$SPECTOK GRAPH=$GRAPH DEVICE=$DEVICE PORT=$PORT ==="
echo "=== method=dspark SERVED=$SERVED P2PACCESS=0 KV=auto NOMTP=$B70_NOMTP ==="
exec bash "$REPO/vllm/w4a8/serve_qwen38_w4a8.sh" start
