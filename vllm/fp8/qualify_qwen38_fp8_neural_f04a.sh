#!/usr/bin/env bash
# F04a forced-rejection oracle using an exact copy of F04's compiled cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F04_ROOT="${F04_ROOT:-$ROOT/results/f04_qwen38_fp8_neural/20260830T012500Z}"
F04_CACHE="${F04_CACHE:-$ROOT/cache/f04_qwen38_fp8_neural/20260830T012500Z/shared}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F04_ROOT/attempt-1/performance.json" \
  "$F04_ROOT/attempt-2/performance.json" \
  "$F04_ROOT/cache-files.sha256"; do
  [ -f "$required" ] || { echo "missing frozen F04 evidence: $required" >&2; exit 1; }
done
[ -d "$F04_CACHE" ] || { echo "missing frozen F04 cache: $F04_CACHE" >&2; exit 1; }

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-serial-f04-local \
  EXPECTED_IMAGE_ID=sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  SPECULATIVE_TOKENS=1 \
  SPECULATIVE_FORCE_REJECT=1 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  CAMPAIGN_ID=f04a \
  CAMPAIGN_LABEL=F04a \
  CONTAINER_PREFIX=qwen38-fp8-neural-f04a-force-reject \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f04a-force-reject.v1 \
  COMPLETION_ROUTE=explicit-work-wait-mtp1-force-reject-f04-cache \
  SHARED_CACHE=1 \
  REQUIRE_REFERENCE_EXACT=1 \
  SEED_CACHE_FROM="$F04_CACHE" \
  SEED_CACHE_MANIFEST="$F04_ROOT/cache-files.sha256" \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-force-reject-p2p0-fp16kv-f04a \
  PUBLISHER_A="$F04_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F04_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f04a_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f04a_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
