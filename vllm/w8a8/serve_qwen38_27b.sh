#!/usr/bin/env bash
# Refreshed vLLM control for the local Qwen3.8-27B compressed-tensors W8A8
# GPTQ artifact. This is a research baseline, not a shelf entry.
#
# The default lane intentionally changes no graph, TP, MTP, prefix-cache, or
# native-kernel factor: TP=2, eager target-only execution, short context. A
# matched TP=1 attempt proved that the 35 GB artifact plus its BF16 LM head
# cannot fit one 31.89 GiB card, so TP=2 is the minimum viable capacity lane.
# The pinned official image contains vLLM 46638857f, torch 2.13.0+xpu, Triton
# XPU 3.7.2, vllm-xpu-kernels 0.1.13.2, Compute Runtime 26.27.39122.11, and
# Level Zero 1.32.0.
#
#   ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh smoke
#
# Raise context, enable graph/TP, and add MTP only as separate qualified arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm/vllm-openai-xpu@sha256:2ac07cf8fde4631de59912f2349729cf130947671b85c087550885cae8e65c46}"
export CKPT="${CKPT:-/models/qwen3.8-27b/w8a8-gptq}"
export SERVED="${SERVED:-qwen3.8-27b-W8A8-gptq}"
export NAME="${NAME:-qwen38_w8a8_vllm_refresh}"
export PORT="${PORT:-18080}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export DEVICE="${DEVICE:-0}"
export GRAPH="${GRAPH:-0}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-4}"
export UTIL="${UTIL:-0.90}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export QUANT="${QUANT:-compressed-tensors}"
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

# The clean-slate runtime root intentionally no longer carries a duplicate of
# the tracked benchmark driver. Keep generated CSVs under the runtime root,
# but execute the source-of-truth script from this repository.
b70_bench() {
  local par="tp${TP}"
  [ "${PP:-1}" -gt 1 ] && par="pp${PP}"
  env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-${par}-eager" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$SCRIPT_DIR/../../bin/35_sweep_bench.sh"
}

b70_dispatch "$@"
