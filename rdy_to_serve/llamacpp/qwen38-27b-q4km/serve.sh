#!/usr/bin/env bash
# Stock Qwen3.8-27B Q4_K_M quality-first daily driver.
# HumanEval+ 164, thinking off, greedy: 0.970 base / 0.927 plus.
# Exact evaluated path: TP=2, F16 KV, 262144 context, MTP off, lab doors off.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NAME="${NAME:-qwen38_stock_q4km_tp2}"
export PORT="${PORT:-18080}"
export SERVED="${SERVED:-hotschmoe-dd}"
export CTX_SIZE="${CTX_SIZE:-262144}"
export BATCH="${BATCH:-1024}"
export UBATCH="${UBATCH:-256}"
export LAB_DOORS="${LAB_DOORS:-0}"
export ENABLE_MTP="${ENABLE_MTP:-0}"

exec bash "$REPO/llamacpp/serve_qwen38_stock_q4km_tp2.sh" "${1:-start}"
