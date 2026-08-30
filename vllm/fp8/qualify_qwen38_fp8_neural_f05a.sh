#!/usr/bin/env bash
# F05a 32K-configured long-context and forced-output MTP1 gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F04B_ROOT="${F04B_ROOT:-$ROOT/results/f04b_qwen38_fp8_neural/20260830T053900Z}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F04B_ROOT/attempt-1/performance.json" \
  "$F04B_ROOT/attempt-2/performance.json"; do
  [ -f "$required" ] || {
    echo "missing frozen F04b MTP1 target evidence: $required" >&2
    exit 1
  }
done

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-serial-f04-local \
  EXPECTED_IMAGE_ID=sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  SPECULATIVE_TOKENS=1 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  MAX_MODEL_LEN=32768 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  EXTRA_WORKLOAD="$SCRIPT_DIR/probe_qwen38_fp8_long_context.py" \
  CAMPAIGN_ID=f05a \
  CAMPAIGN_LABEL=F05a \
  CONTAINER_PREFIX=qwen38-fp8-neural-f05a-mtp1-32k \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f05a-mtp1-32k.v1 \
  COMPLETION_ROUTE=explicit-work-wait-mtp1-packed-rms-deterministic-reduction-32k \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p0-fp16kv-f05a-32k \
  PUBLISHER_A="$F04B_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F04B_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f05a_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f05a_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
