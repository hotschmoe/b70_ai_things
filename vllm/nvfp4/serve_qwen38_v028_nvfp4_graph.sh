#!/usr/bin/env bash
# Qualified vLLM 0.28 Qwen3.8 NVFP4 TP1 breakable-graph capacity candidate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TP="${TP:-1}"
export DEVICE="${DEVICE:-0}"
export BREAKABLE="${BREAKABLE:-1}"
export GRAPH="${GRAPH:-1}"
export CGMODE="${CGMODE:-PIECEWISE}"
export CAPSIZES="${CAPSIZES:-1,2,4}"
export COMPILESZ="${COMPILESZ-}"
export IGP="${IGP:-false}"
export MAXSEQS="${MAXSEQS:-4}"
export MAXLEN="${MAXLEN:-4096}"
export UTIL="${UTIL:-0.92}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export B70_NVFP4_F8_SCALE_M_MAX="${B70_NVFP4_F8_SCALE_M_MAX:-8}"

exec bash "$SCRIPT_DIR/serve_qwen38_v028_nvfp4.sh" "$@"
