#!/usr/bin/env bash
# Verified Qwen3.8-27B-OBLITERATED V3 Q4_K_M daily-driver configuration.
#
# Topology: two independent one-card llama.cpp replicas behind nginx :18080.
# Each replica has one 245760-token slot with Q8_0 KV. The V3 GGUF's embedded
# MTP head is enabled with draft max 3. Served id: hotschmoe-dd.
#
# Measured 2026-08-23:
#   no MTP: 23.84 + 23.85 = 47.69 tok/s aggregate
#   MTP3:   40.35 + 41.51 = 81.86 tok/s aggregate
#   MTP3 soak: 338/338 coherent, zero degenerate/errors, c4 for 300 seconds
#
#   ./bin/gpu-run bash rdy_to_serve/llamacpp/qwen38-27b-obliterated-q4km/serve.sh start
#   ./bin/gpu-run bash rdy_to_serve/llamacpp/qwen38-27b-obliterated-q4km/serve.sh stop
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NAME="${NAME:-qwen38_oblit_q4km_dp2}"
export PORT="${PORT:-18080}"
export P0="${P0:-18181}"
export P1="${P1:-18182}"
export SERVED="${SERVED:-hotschmoe-dd}"
export CTX_SIZE="${CTX_SIZE:-245760}"
export PARALLEL="${PARALLEL:-1}"
export BATCH="${BATCH:-1024}"
export UBATCH="${UBATCH:-256}"
export KV_TYPE="${KV_TYPE:-q8_0}"
export LAB_DOORS="${LAB_DOORS:-1}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export MTP_SIDECAR="${MTP_SIDECAR:-0}"
export MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-3}"

exec bash "$REPO/llamacpp/serve_qwen38_obliterated_q4km_dp2.sh" "${1:-start}"
