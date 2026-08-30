#!/usr/bin/env bash
# F06f: full corrected-kernel MTP1 direct-P2P qualification.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F05D_ROOT="${F05D_ROOT:-$ROOT/results/f05d_qwen38_fp8_neural/20260830T075200Z}"
F06E_STAMP="${F06E_STAMP:-20260830T124200Z}"
F06E_ROOT="${F06E_ROOT:-$ROOT/results/f06e_qwen38_fp8_neural_p2p/$F06E_STAMP}"
F06E_CACHE="${F06E_CACHE:-$ROOT/cache/f06e_qwen38_fp8_neural_p2p/$F06E_STAMP}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F05D_ROOT/attempt-1/performance.json" \
  "$F05D_ROOT/attempt-2/performance.json" \
  "$F06E_ROOT/verdict.txt" "$F06E_ROOT/concurrent-quality.json" "$F06E_CACHE"; do
  [ -e "$required" ] || { echo "missing frozen F06f input: $required" >&2; exit 1; }
done
grep -Fq 'VERDICT -> F06e PASS:' "$F06E_ROOT/verdict.txt" || {
  echo "frozen F06e evidence is not a pass" >&2
  exit 1
}

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local \
  EXPECTED_IMAGE_ID=sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81 \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  P2P_ACCESS=1 \
  SPECULATIVE_TOKENS=1 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  MAX_MODEL_LEN=32768 \
  MAX_NUM_SEQS=4 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  F05B_OUTPUT_TOKENS=512 \
  F05B_BATCHES=2 \
  EXTRA_WORKLOAD="$SCRIPT_DIR/probe_qwen38_fp8_concurrent_gate.py" \
  EXTRA_WORKLOAD_RESULT=concurrent-gate.json \
  CONCURRENT_QUALIFIED=1 \
  CAMPAIGN_ID=f06f \
  CAMPAIGN_LABEL=F06f \
  CONTAINER_PREFIX=qwen38-fp8-neural-f06f-mixed-gdn-p2p1-c4 \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f06f-mixed-gdn-p2p1-c4.v1 \
  COMPLETION_ROUTE=mtp1-packed-rms-deterministic-mixed-gdn-p2p1-c4 \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=1 \
  SEED_CACHE_FROM="$F06E_CACHE" \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f06f-32k-c4 \
  PUBLISHER_A="$F05D_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F05D_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f06f_qwen38_fp8_neural_p2p/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f06f_qwen38_fp8_neural_p2p/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
