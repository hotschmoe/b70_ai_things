#!/usr/bin/env bash
# 119_serve_push_ar_graph.sh -- 27B-W8A8 TP=2 GRAPH=1 serve with the CAPTURABLE push all-reduce (decode too).
# This is the K.6 production target: PUSH_AR_GRAPH=1 + PUSH_AR_MIN_NUMEL=0 -> EVERY all-reduce uses the push
# transport, decode included, via the device-side L0-event sync that records into torch's XPUGraph (so decode
# is no longer on oneCCL). Mirrors 108 but: builds libxpu_push_ar_graph.so (scripts/118), GRAPH=1, graph gate on.
#
# Usage:
#   B70_AUTO_RESET=1 ./bin/gpu-run bash scripts/119_serve_push_ar_graph.sh smoke   # coherence gate (push-graph ON)
#   PUSH_AR_DISABLE=1 ./bin/gpu-run bash scripts/119_serve_push_ar_graph.sh smoke  # oneCCL baseline (decode)
#   ./bin/gpu-run bash scripts/119_serve_push_ar_graph.sh run                       # + bench
#   bash scripts/119_serve_push_ar_graph.sh stop
# WEDGE NOTE (AGENTS.md): GRAPH=1 TP=2 capture -- run ONE at a time, watch for EMPTY output (broken capture),
# reboot-only recovery if it wedges. Guard (lib.sh) does pre-flight probe + graceful teardown + post-probe.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp"
export IMG="${IMG:-vllm-xpu-env:int8g}"

# ---- build the CAPTURABLE .so in the serve image (ABI match) if missing / REBUILD_SO=1 ----
SO_HOST="$ROOT/push_ar/libxpu_push_ar_graph.so"
if [ "${PUSH_AR_DISABLE:-0}" != "1" ] && [ ! -f "$SO_HOST" -o "${REBUILD_SO:-0}" = "1" ]; then
  echo "=== building capturable push_ar .so in $IMG ==="
  mkdir -p "$ROOT/push_ar"
  docker run --rm -v "$REPO:$REPO" -v "$ROOT/push_ar:$ROOT/push_ar" --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$REPO/scripts/118_xpu_push_ar_graph.cpp' \
      -o '$SO_HOST' -lze_loader -lrt && echo 'SO BUILD OK'
  " || { echo '[!] .so build failed'; exit 1; }
fi

# ---- replicate the shelf serve env (GRAPH=1 production capture) ----
export CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
export SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-mtp-pushar-graph}"
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"                    # PRODUCTION capture (the whole point)
export IGP="${IGP:-false}"
export NOMM="${NOMM:-1}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export MTPTOK="${MTPTOK:-3}"
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"
export COMPILESZ="${COMPILESZ-}"
export P2PACCESS="${P2PACCESS:-0}"
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
DOCKER_ENV=( -e PYTHONPATH=/opt/push_ar:/opt/mtp_shim
             -e PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py
             -e PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so
             -e PUSH_AR_DISABLE="${PUSH_AR_DISABLE:-0}"
             -e PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-1}"
             -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-0}" )

echo "=== push-ar GRAPH A/B :: DISABLE=${PUSH_AR_DISABLE:-0} GRAPH=$GRAPH PUSH_AR_GRAPH=${PUSH_AR_GRAPH:-1} MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-0} P2PACCESS=$P2PACCESS IMG=$IMG ==="
source "$SHELF/../_common/lib.sh"
b70_dispatch "$@"
