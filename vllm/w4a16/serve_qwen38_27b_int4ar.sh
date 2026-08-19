#!/usr/bin/env bash
# Qwen3.8-27B AutoRound INT4 (W4A16) -- S2 speed+quality vehicle.
# Default image is the newest public vLLM XPU nightly we already pulled
# (SergiioB 0.27.2rc1 digest f01e24f6). Override IMG to use another
# digest, a home-built tag, or a Steve-stack rebuild. Not int8g-v0260
# unless you set it.
#
#   TP=1 GRAPH=0 PORT=18080 NAME=qwen38_int4ar \
#     SERVED=qwen3.8-27b-W4A16-autoround-mtp5 \
#     ./bin/gpu-run --card 0 bash vllm/w4a16/serve_qwen38_27b_int4ar.sh start
#   bash vllm/w4a16/serve_qwen38_27b_int4ar.sh stop
# LOOP 27: TP=2 on this digest is D10 (2021.15 device_fd; 2021.17 is SYCL-8).
# GRAPH=1 G1 garbage (D11). Gated path is TP=1 GRAPH=0. Isolated TRITON
# cache required (0.26 cache is libsycl.so.8; this image is .so.9).
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
# intel/vllm:0.21.0-xpu is SYCL-8 + in-image 2021.17. lib.sh --entrypoint vllm
# skips setvars (torch then misses libccl). Bind this wrapper over PATH vllm.
case "${IMG}" in
  intel/vllm*|*/intel/vllm*)
    WRAP="$SCRIPT_DIR/intel021_vllm_entrypoint.sh"
    chmod +x "$WRAP"
    MOUNTS+=( -v "$WRAP:/opt/venv/bin/vllm:ro" )
    echo "=== intel/vllm wrapper -> setvars + CCL 2021.17 (SYCL-8) ===" >&2
    ;;
esac
if [ -n "${XPU_C_SO:-}" ]; then
  MOUNTS+=( -v "$XPU_C_SO:$PKGD/_xpu_C.abi3.so:ro" )
  echo "=== overlay _xpu_C <- $XPU_C_SO ===" >&2
fi
if [ -n "${GDN_LIB:-}" ]; then
  MOUNTS+=( -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  echo "=== overlay gdn lib <- $GDN_LIB ===" >&2
fi

# PRE.10: 0.27 nightlies ship oneCCL 2021.15 in /opt/venv/lib and die at
# TP>1 (ze mem_to_ipc_handle: device_fd is invalid). Host 2021.17 is
# SYCL-8 and ImportErrors on this SYCL-9 nightly. Default OFF. Set
# CCL217=/mnt/vm_8tb/b70/ccl_2021.17/2021.17 only on a matching-ABI image.
CCL217="${CCL217-}"
if [ -n "$CCL217" ] && [ -d "$CCL217/lib" ]; then
  SYCL8SHIM="${SYCL8SHIM:-/mnt/vm_8tb/b70/qwen38-w8a8-dspark/sycl8shim}"
  mkdir -p "$SYCL8SHIM"
  ln -sfn /opt/venv/lib/libsycl.so.9 "$SYCL8SHIM/libsycl.so.8"
  # Torch 2.13 libtorch_xpu is DT_RPATH $ORIGIN/../../../.. so LD_LIBRARY_PATH
  # cannot win. Bind the 2021.17 objects over the nightly's 2021.15 copies.
  MOUNTS+=( -v "$CCL217:/opt/ccl217:ro" )
  MOUNTS+=( -v "$CCL217/lib/libccl.so.1.0:/opt/venv/lib/libccl.so:ro" )
  MOUNTS+=( -v "$CCL217/lib/libccl.so.1.0:/opt/venv/lib/libccl.so.1:ro" )
  MOUNTS+=( -v "$CCL217/lib/libccl.so.1.0:/opt/venv/lib/libccl.so.1.0:ro" )
  MOUNTS+=( -v "$CCL217/lib/libccl.so.2.0:/opt/venv/lib/libccl.so.2:ro" )
  MOUNTS+=( -v "$CCL217/lib/libccl.so.2.0:/opt/venv/lib/libccl.so.2.0:ro" )
  MOUNTS+=( -v "$SYCL8SHIM:/opt/sycl8shim:ro" )
  DOCKER_ENV+=( -e "CCL_ROOT=/opt/ccl217" )
  DOCKER_ENV+=( -e "LD_LIBRARY_PATH=/opt/sycl8shim:/opt/venv/lib:/tmp/ucx_install/lib:/usr/local/lib" )
  echo "=== overlay oneCCL 2021.17 FILES over /opt/venv/lib/libccl.so* (+ sycl8 shim) ===" >&2
fi

source "$REPO/rdy_to_serve/_common/lib.sh"
b70_dispatch "$@"
