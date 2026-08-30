#!/usr/bin/env bash
# F05b C4 serial/concurrent raw-token coherence gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F05A_ROOT="${F05A_ROOT:-$ROOT/results/f05a_qwen38_fp8_neural/20260830T061000Z}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F05A_ROOT/attempt-1/performance.json" \
  "$F05A_ROOT/attempt-2/performance.json"; do
  [ -f "$required" ] || {
    echo "missing frozen F05a target evidence: $required" >&2
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
  MAX_NUM_SEQS=4 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  EXTRA_WORKLOAD="$SCRIPT_DIR/probe_qwen38_fp8_concurrent.py" \
  EXTRA_WORKLOAD_RESULT=concurrent.json \
  CAMPAIGN_ID=f05b \
  CAMPAIGN_LABEL=F05b \
  CONTAINER_PREFIX=qwen38-fp8-neural-f05b-mtp1-c4 \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f05b-mtp1-c4.v1 \
  COMPLETION_ROUTE=explicit-work-wait-mtp1-packed-rms-deterministic-reduction-c4 \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p0-fp16kv-f05b-32k-c4 \
  PUBLISHER_A="$F05A_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F05A_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f05b_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f05b_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
