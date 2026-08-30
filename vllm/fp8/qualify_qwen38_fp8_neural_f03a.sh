#!/usr/bin/env bash
# F03a two-fresh-process control sharing lifetime 1's compiler cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F02_ROOT="${F02_ROOT:-$ROOT/results/f02_qwen38_fp8_neural/20260829T231100Z}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F02_ROOT/attempt-1/performance.json" \
  "$F02_ROOT/attempt-2/performance.json"; do
  [ -f "$required" ] || { echo "missing F02 Work.wait reference: $required" >&2; exit 1; }
done

exec env \
  CAMPAIGN_ID=f03a \
  CAMPAIGN_LABEL=F03a \
  CONTAINER_PREFIX=qwen38-fp8-neural-f03a-shared-cache \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f03a-shared-cache.v1 \
  COMPLETION_ROUTE=explicit-work-wait-shared-cache \
  SHARED_CACHE=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f03a-shared-cache \
  PUBLISHER_A="$F02_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F02_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f03a_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f03a_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
