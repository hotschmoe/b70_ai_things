#!/usr/bin/env bash
# F10a: closest runnable r32 reproduction, including its 9/12 GiB cgroup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUBLISHER_A="$SOURCE/experiments/qwen38-27b-b70/data/qwen38-fp8-mtp1-deterministic-r32a/performance.json"
PUBLISHER_B="$SOURCE/experiments/qwen38-27b-b70/data/qwen38-fp8-mtp1-deterministic-r32b/performance.json"

for required in "$PUBLISHER_A" "$PUBLISHER_B"; do
  [ -f "$required" ] || { echo "missing publisher reference: $required" >&2; exit 1; }
done

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local \
  EXPECTED_IMAGE_ID=sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81 \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  P2P_ACCESS=1 \
  SPECULATIVE_TOKENS=1 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  COMPILATION_PROFILE=publisher \
  GPU_MEMORY_UTILIZATION=0.96 \
  MAX_MODEL_LEN=1024 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  MEMORY_GIB=9 \
  MEMORY_SWAP_GIB=12 \
  MAX_SWAP_USED_MIB=3072 \
  CAMPAIGN_ID=f10a \
  CAMPAIGN_LABEL=F10a \
  CONTAINER_PREFIX=qwen38-fp8-neural-f10a-publisher-cgroup \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f10a-publisher-cgroup.v1 \
  COMPLETION_ROUTE=mtp1-packed-rms-publisher-compile-p2p1-cgroup9-12 \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=0 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f10a-r32-cgroup \
  PUBLISHER_A="$PUBLISHER_A" \
  PUBLISHER_B="$PUBLISHER_B" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f10a_qwen38_fp8_neural_publisher_cgroup/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f10a_qwen38_fp8_neural_publisher_cgroup/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
