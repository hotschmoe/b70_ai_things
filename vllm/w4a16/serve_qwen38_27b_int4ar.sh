#!/usr/bin/env bash
# Qwen3.8-27B AutoRound INT4 (W4A16) -- S2 speed+quality vehicle.
# Default image is the newest public vLLM XPU nightly we already pulled
# (SergiioB 0.27.2rc1 digest f01e24f6). Override IMG to use another
# digest, a home-built tag, or a Steve-stack rebuild. Not int8g-v0260
# unless you set it.
#
#   TP=2 MTPTOK=5 GRAPH=1 PORT=18080 NAME=qwen38_int4ar \
#     SERVED=qwen3.8-27b-W4A16-autoround-mtp5 \
#     ./bin/gpu-run bash vllm/w4a16/serve_qwen38_27b_int4ar.sh start
#   bash vllm/w4a16/serve_qwen38_27b_int4ar.sh stop
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMG="${IMG:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
export CKPT="${CKPT:-/models/qwen3.8-27b/int4-autoround}"
export SERVED="${SERVED:-qwen3.8-27b-W4A16-autoround-mtp5}"
export NAME="${NAME:-qwen38_int4ar}"
export PORT="${PORT:-18080}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"
export CGMODE="${CGMODE:-PIECEWISE}"
export DTYPE="${DTYPE:-float16}"
export UTIL="${UTIL:-0.88}"
export MAXLEN="${MAXLEN:-16384}"
export MAXSEQS="${MAXSEQS:-8}"
export CAPSIZES="${CAPSIZES:-1,2,4,5,6,8}"
export MTPTOK="${MTPTOK:-5}"
export NOMM="${NOMM:-1}"
export TOOLCALL="${TOOLCALL:-1}"
export TOOLPARSER="${TOOLPARSER:-qwen3_xml}"
export REASONPARSER="${REASONPARSER:-qwen3}"
# Let vLLM infer auto-round -> INC. Set QUANT=inc if a given image needs it.
export QUANT="${QUANT:-}"
export EXTRA_ARGS="${EXTRA_ARGS:---language-model-only}"

DOCKER_ENV=()
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> ${B70_EXTRA_ENV} ===" >&2
fi

# Optional Steve-kernel / FA overlays (host paths -> container).
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
MOUNTS=()
if [ -n "${XPU_C_SO:-}" ]; then
  MOUNTS+=( -v "$XPU_C_SO:$PKGD/_xpu_C.abi3.so:ro" )
  echo "=== overlay _xpu_C <- $XPU_C_SO ===" >&2
fi
if [ -n "${GDN_LIB:-}" ]; then
  MOUNTS+=( -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  echo "=== overlay gdn lib <- $GDN_LIB ===" >&2
fi

source "$REPO/rdy_to_serve/_common/lib.sh"
b70_dispatch "$@"
