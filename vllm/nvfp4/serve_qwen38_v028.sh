#!/usr/bin/env bash
# Stock vLLM 0.28 XPU control for RadixArk Qwen3.8-27B NVFP4.
#
# This lane deliberately mounts no pre-refresh ABI library or Python shim. It
# establishes what the exact official release supports before the retained
# NVFP4 source is ported to the current vllm-xpu-kernels layout.
#
#   ./bin/gpu-run bash vllm/nvfp4/serve_qwen38_v028.sh smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm/vllm-openai-xpu@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3}"
export CKPT="${CKPT:-/models/qwen3.8-27b/nvfp4-radixark}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-radixark-vllm028-stock}"
export NAME="${NAME:-qwen38_nvfp4_v028_stock}"
export PORT="${PORT:-18080}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export GRAPH="${GRAPH:-0}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-4}"
export UTIL="${UTIL:-0.90}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export NOMM="${NOMM:-1}"
export REASONPARSER="${REASONPARSER:-qwen3}"
export IN="${IN:-512}"
export OUT="${OUT:-512}"
export CONC="${CONC:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:---language-model-only --generation-config vllm --no-async-scheduling --uvicorn-log-level warning}"

HOST_CKPT="$(cd "$SCRIPT_DIR/../.." && pwd)/models/files/${CKPT#/models/}"
[ -d "$HOST_CKPT" ] || {
  echo "Missing checkpoint: $HOST_CKPT" >&2
  exit 1
}

MOUNTS=()
DOCKER_ENV=()

source "$SCRIPT_DIR/../../rdy_to_serve/_common/lib.sh"

b70_bench() {
  local par="tp${TP}"
  [ "${PP:-1}" -gt 1 ] && par="pp${PP}"
  env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-${par}$([ "$GRAPH" = 1 ] && echo -graph)" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$SCRIPT_DIR/../../bin/35_sweep_bench.sh"
}

b70_dispatch "$@"
