#!/usr/bin/env bash
# Qwen3.8-27B NVFP4. Default is Inferact ModelOpt (official vLLM recipe,
# quant_algo=NVFP4). Unsloth CT-mixed is MODEL_REL=qwen3.8-27b/nvfp4-unsloth.
# Thin wrapper over vllm/nvfp4/serve_nvfp4_27b.sh via MODEL_REL.
#
#   TP=2 ./bin/gpu-run bash serve.sh start     # Inferact TP=2 GRAPH+MTP5 @200k
#   bash serve.sh stop
#
# Gated 2026-08-15m (Inferact, TP=2): fused GRAPH + MTP5 + prefix + push-AR,
#   KV_FP8=0, MAXLEN=200000. kv_gate 3/3, 18/18 PASS, c4 stayed up.
#   Code c1 29.0 / c4 agg 76.4. NOT a DD vs 3.6 NVFP4 TP=2 48.9 / 103.
# Unsloth one-card research: call serve_nvfp4_27b.sh DIRECTLY with MTPTOK=
#   (this wrapper's TP=1 MTPTOK:-5 treats empty as 5).
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
export MODEL_REL="${MODEL_REL:-qwen3.8-27b/nvfp4-modelopt}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-modelopt}"

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
  export TP MAXLEN="${MAXLEN:-200000}" MTPTOK="${MTPTOK-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
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
