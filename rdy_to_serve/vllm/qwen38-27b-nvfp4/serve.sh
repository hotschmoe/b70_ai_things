#!/usr/bin/env bash
# Qwen3.8-27B NVFP4 (unsloth/Qwen3.8-27B-NVFP4, compressed-tensors mixed-precision).
# Thin wrapper over vllm/nvfp4/serve_nvfp4_27b.sh via MODEL_REL.
#
#   TP=2 ./bin/gpu-run bash serve.sh start     # TP=2 native 262144, bf16 KV
#   ./bin/gpu-run --card 0 bash serve.sh start # single-card (wrapper default 100k+MTP5, UNMEASURED)
#   bash serve.sh stop
#
# Measured one-card research recipe (2026-08-15e; FITS, NOT coherent):
#   call serve_nvfp4_27b.sh DIRECTLY with MTPTOK= (empty). This wrapper's TP=1
#   `MTPTOK:-5` treats empty as 5. Do not use this wrapper for the one-card
#   isolation serve.
#     TP=1 CARD=0 PORT=8078 NAME=unsloth_c0 MODE=fused GRAPH=0 \
#       MAXLEN=8192 UTIL=0.90 MAXSEQS=4 PREFIXCACHE=0 \
#       IMG=vllm-xpu-env:int8g-v0260 \
#       MODEL_REL=qwen3.8-27b/nvfp4-unsloth \
#       SERVED=qwen3.8-27b-NVFP4-unsloth MTPTOK= REASONPARSER=qwen3 \
#       ./bin/gpu-run --card 0 bash vllm/nvfp4/serve_nvfp4_27b.sh
#   Load 24.71 GiB, KV 1.89 GiB / 35k tok. 2026-08-15f: COHERENT after
#   sitecustomize (1e) tiled channel-FP8 dequant (Paris / 43). Slow -- do
#   not promote. Inferact ModelOpt is the fast coherent 3.8 NVFP4.
# Inferact ModelOpt (coherent): MODEL_REL=qwen3.8-27b/nvfp4-modelopt.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

NAME="${NAME:-nvfp4_38_27b}"
ACTION="${1:-start}"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

TP="${TP:-2}"
export MODE=fused GRAPH=1 PORT="${PORT:-8078}" NAME="$NAME"
export MODEL_REL="${MODEL_REL:-qwen3.8-27b/nvfp4-unsloth}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-unsloth}"

if [ "$TP" = 1 ]; then
  export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
  export CARD="${CARD:-0}" MAXLEN="${MAXLEN:-100352}" UTIL="${UTIL:-0.95}" \
         MAXSEQS="${MAXSEQS:-8}" CAPSIZES="${CAPSIZES:-1,2,4,8}" MAXBATCH="${MAXBATCH:-2048}" \
         MTPTOK="${MTPTOK:-5}"
  export KV_FP8="${KV_FP8:-0}" KV_SCALES="${KV_SCALES-}"
  export PREFIXCACHE="${PREFIXCACHE:-1}"
  export FUSED_SO="${FUSED_SO:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so}"
  export GDN_LIB="${GDN_LIB:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  for _pcf in B70_EMBED_INT8=1 B70_PC_EAGLE_KEEP=1 B70_PC_CHUNK_ALIGN=1 \
      B70_NVFP4_F8_SCALE_M_MAX=8; do
    case " ${B70_EXTRA_ENV:-} " in
      *" ${_pcf%%=*}="*) : ;;
      *) export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }$_pcf" ;;
    esac
  done
else
  export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
  # MTPTOK:- would treat a caller-empty value as 5. Use - so MTPTOK= disables spec.
  export TP MAXLEN="${MAXLEN:-262144}" MTPTOK="${MTPTOK-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
         UTIL="${UTIL:-0.85}" MAXSEQS="${MAXSEQS:-8}" MAXBATCH="${MAXBATCH:-16384}"
  export PUSH_AR="${PUSH_AR:-1}" PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-1}" \
         PUSH_AR_MAXB="${PUSH_AR_MAXB:-268435456}" PREFIXCACHE="${PREFIXCACHE:-1}"
  export KV_FP8="${KV_FP8:-0}"
  if [ "$KV_FP8" = 0 ]; then
    export KV_SCALES="${KV_SCALES-}"
  else
    export KV_SCALES="${KV_SCALES:-}"
  fi
  export FUSED_SO="${FUSED_SO:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so}"
  export GDN_LIB="${GDN_LIB:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  for _pcf in B70_PC_EAGLE_KEEP=1 B70_PC_CHUNK_ALIGN=1 B70_NVFP4_F8_SCALE_M_MAX=8; do
    case " ${B70_EXTRA_ENV:-} " in
      *" ${_pcf%%=*}="*) : ;;
      *) export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }$_pcf" ;;
    esac
  done
fi

exec bash "$REPO/vllm/nvfp4/serve_nvfp4_27b.sh"
