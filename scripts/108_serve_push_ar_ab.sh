#!/usr/bin/env bash
# 108_serve_push_ar_ab.sh -- 27B-W8A8 TP=2 serve with the hand-rolled PUSH all-reduce (contrib/
# vllm_push_allreduce) swapped in for oneCCL, A/B vs the stock oneCCL path. STANDALONE research script:
# it reuses the shared rdy_to_serve/_common/lib.sh and REPLICATES the unedited
# rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh env -- it does NOT edit any rdy_to_serve file.
#
# Differences from the shelf recipe:
#   - mounts contrib/vllm_push_allreduce at /opt/push_ar (FIRST on PYTHONPATH) + the built .so;
#     my sitecustomize chains the MTP shim's sitecustomize (PUSH_AR_CHAIN_SITECUSTOMIZE) so BOTH run.
#   - GRAPH defaults 0 (EAGER): the push op has a host barrier -> not SYCL-graph-capturable.
#   - P2PACCESS=0 (our op's P2P is L0-IPC, independent of oneCCL -> oneCCL warmup stays host-staged, no H.13).
#   - PUSH_AR_DISABLE=1 gives the oneCCL baseline (same eager serve) for the A/B.
#
# Usage:
#   ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke         # coherence-gated probe (push ON)
#   PUSH_AR_DISABLE=1 ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke   # baseline (oneCCL)
#   ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh run           # + bench (A side)
#   bash scripts/108_serve_push_ar_ab.sh stop
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp"

# ---- build libxpu_push_ar_torch.so in the SERVE image (ABI match with int8g's torch) if missing ----
export IMG="${IMG:-vllm-xpu-env:int8g}"
SO_HOST="$ROOT/push_ar/libxpu_push_ar_torch.so"
if [ "${PUSH_AR_DISABLE:-0}" != "1" ] && [ ! -f "$SO_HOST" -o "${REBUILD_SO:-0}" = "1" ]; then
  echo "=== building push_ar .so in $IMG ==="
  mkdir -p "$ROOT/push_ar"
  docker run --rm -v "$REPO:$REPO" -v "$ROOT/push_ar:$ROOT/push_ar" --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$REPO/scripts/106_xpu_push_ar_torch.cpp' \
      -o '$SO_HOST' -lze_loader -lrt && echo 'SO BUILD OK'
  " || { echo '[!] .so build failed'; exit 1; }
fi

# ---- replicate the shelf serve env (unedited shelf is the source of truth for these) ----
export CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
export SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-mtp-pushar}"   # distinct id: encodes the push-ar variant
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-0}"                    # EAGER for push-ar capturability (shelf uses 1)
export IGP="${IGP:-false}"
export NOMM="${NOMM:-1}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export MTPTOK="${MTPTOK:-3}"
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"
export COMPILESZ="${COMPILESZ-}"
export P2PACCESS="${P2PACCESS:-0}"            # our P2P is L0-IPC, independent of this -> keep oneCCL host-staged
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
PUSH_AR_DIR="$REPO/contrib/vllm_push_allreduce"

MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SHELF/patches:/opt/mtp_shim:ro"
         -v "$PUSH_AR_DIR:/opt/push_ar:ro"
         -v "$SO_HOST:/opt/push_ar_so/libxpu_push_ar_torch.so:ro" )
# /opt/push_ar FIRST -> my sitecustomize loads, chains the MTP shim's, then patches all_reduce.
DOCKER_ENV=( -e PYTHONPATH=/opt/push_ar:/opt/mtp_shim
             -e PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py
             -e PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so
             -e PUSH_AR_DISABLE="${PUSH_AR_DISABLE:-0}"
             -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-0}" )  # >0 = capture-gated (prefill-only) mode

echo "=== push-ar A/B :: PUSH_AR_DISABLE=${PUSH_AR_DISABLE:-0} GRAPH=$GRAPH P2PACCESS=$P2PACCESS IMG=$IMG ==="
source "$SHELF/../_common/lib.sh"
b70_dispatch "$@"
