#!/usr/bin/env bash
# Qwen3.8 NVFP4 TP2 FULL-graph oracle: Triton attention and true pidfd IPC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"
export BREAKABLE="${BREAKABLE:-0}"
export BREAKABLE_AR="${BREAKABLE_AR:-0}"
export CGMODE="${CGMODE:-FULL}"
export CAPSIZES="${CAPSIZES:-1}"
export COMPILESZ="${COMPILESZ:-1}"
export IGP="${IGP:-true}"
export MAXSEQS="${MAXSEQS:-1}"
export MAXLEN="${MAXLEN:-4096}"
export UTIL="${UTIL:-0.85}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export SYCLKERNELS="${SYCLKERNELS:-1}"
export P2PACCESS="${P2PACCESS:-0}"
export IPCX="${IPCX:-pidfd}"
export PORT="${PORT:-18182}"
export NAME="${NAME:-qwen38_nvfp4_v028_full_tp2}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-radixark-vllm028-full-tp2}"
export B70_NVFP4_F8_SCALE_M_MAX="${B70_NVFP4_F8_SCALE_M_MAX:-8}"
export EXTRA_ARGS="${EXTRA_ARGS:---language-model-only --generation-config vllm --no-async-scheduling --uvicorn-log-level warning --attention-backend TRITON_ATTN}"

exec bash "$SCRIPT_DIR/serve_qwen38_v028_nvfp4.sh" "$@"
