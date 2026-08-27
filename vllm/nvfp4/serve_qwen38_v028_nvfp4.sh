#!/usr/bin/env bash
# vLLM 0.28 Qwen3.8 NVFP4 candidate with the source-built XPU W4A16 op.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

export IMG="${IMG:-vllm/vllm-openai-xpu@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3}"
export CKPT="${CKPT:-/models/qwen3.8-27b/nvfp4-radixark}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-radixark-vllm028-onednn}"
export NAME="${NAME:-qwen38_nvfp4_v028_onednn}"
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

NVFP4_SO="${NVFP4_SO:-/mnt/vm_8tb/b70/nvfp4_kernel_v028/_xpu_C.abi3.so}"
GDN_SO="${GDN_SO:-/mnt/vm_8tb/b70/nvfp4_kernel_v028/libgdn_attn_kernels_xe_2.so}"
test -f "$NVFP4_SO" || {
  echo "Missing source-built NVFP4 extension: $NVFP4_SO" >&2
  exit 1
}
test -f "$GDN_SO" || {
  echo "Missing source-built GDN sidecar: $GDN_SO" >&2
  exit 1
}

HOST_CKPT="$REPO/models/files/${CKPT#/models/}"
test -d "$HOST_CKPT" || {
  echo "Missing checkpoint: $HOST_CKPT" >&2
  exit 1
}

MOUNTS=(
  -v "$NVFP4_SO:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so:ro"
  -v "$GDN_SO:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so:ro"
  -v "$SCRIPT_DIR/v028_patch:/b70_nvfp4_v028_patch:ro"
)
DOCKER_ENV=(
  -e PYTHONPATH=/b70_nvfp4_v028_patch
  -e B70_NVFP4_V028=1
  -e B70_NVFP4_F8_SCALE_M_MAX="${B70_NVFP4_F8_SCALE_M_MAX:-8}"
)
if [ "${BREAKABLE:-0}" = 1 ]; then
  DOCKER_ENV+=(
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
    -e VLLM_USE_AOT_COMPILE=0
  )
fi

source "$REPO/rdy_to_serve/_common/lib.sh"

b70_bench() {
  local par="tp${TP}"
  test "${PP:-1}" -gt 1 && par="pp${PP}"
  env NAME="$NAME" MODEL="$SERVED" \
    LABEL="${SERVED}-${par}$([ "$GRAPH" = 1 ] && echo -graph)" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$REPO/bin/35_sweep_bench.sh"
}

b70_dispatch "$@"
