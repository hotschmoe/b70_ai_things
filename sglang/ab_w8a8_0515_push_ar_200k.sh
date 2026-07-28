#!/usr/bin/env bash
# Controlled sglang 0.5.15 W8A8 200K/BF16-KV push-prefill A/B.
#
# Both arms use the proven one-request state sizing and run identity, physical
# capacity, 18-stream coherence, unique cold prefill, exact 190K cold/warm
# retrieval, cache metrics, fatal-log, teardown, and card-health gates.
#
# Caller must hold both cards:
#   ./bin/gpu-run bash sglang/ab_w8a8_0515_push_ar_200k.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUALIFY="$REPO/sglang/qualify_w8a8_0515.sh"
PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-1048576}"

run_arm() {
  local label="$1"
  local enabled="$2"
  local name="$3"
  local served="$4"

  echo "ARM -> $label push_ar=$enabled min_numel=$PUSH_AR_MIN_NUMEL"
  PUSH_AR="$enabled" \
    PUSH_AR_MIN_NUMEL="$PUSH_AR_MIN_NUMEL" \
    NAME="$name" \
    SERVED="$served" \
    MAXLEN=200000 \
    MAXREQ=1 \
    MAMBA_CACHE=4 \
    MIN_POOL_TOKENS=190128 \
    RUN_COHERENCE=1 \
    RUN_PERF=0 \
    RUN_PREFILL=1 \
    RUN_NEEDLE=1 \
    bash "$QUALIFY"
}

run_arm \
  "sglang-0.5.15-200k-prefill-ar-off" \
  0 \
  "sglang_w8a8_0515_200k_ar_off" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515-200k-ar-off"

run_arm \
  "sglang-0.5.15-200k-prefill-ar-on" \
  1 \
  "sglang_w8a8_0515_200k_ar_on" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515-200k-ar-on"

echo "VERDICT -> both sglang 0.5.15 W8A8 200K push-prefill A/B arms completed"
