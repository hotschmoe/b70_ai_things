#!/usr/bin/env bash
# F03 source-default collective completion using the complete F02 protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F02_ROOT="${F02_ROOT:-$ROOT/results/f02_qwen38_fp8_neural/20260829T231100Z}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F02_ROOT/attempt-1/performance.json" \
  "$F02_ROOT/attempt-2/performance.json"; do
  [ -f "$required" ] || { echo "missing F02 work-wait reference: $required" >&2; exit 1; }
done

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-source-default-communicator-f03 \
  EXPECTED_IMAGE_ID=sha256:c4fc0d651aedd8088daaf57d5de9f623f68f9066a36956fd67652d472c18c3d0 \
  COMMUNICATOR_SHA256=527cbfb250760abc62096ee7cd612307b821f21b72dee1687ad866620ec89b6d \
  CAMPAIGN_ID=f03 \
  CAMPAIGN_LABEL=F03 \
  CONTAINER_PREFIX=qwen38-fp8-neural-f03-source-default \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f03-source-default.v1 \
  COMPLETION_ROUTE=source-default-sync \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f03-source-default \
  PUBLISHER_A="$F02_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F02_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f03_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f03_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
