#!/usr/bin/env bash
# 110_serve_pp2_graph_mtp.sh -- PP=2 in the PRODUCTION config (GRAPH=1 + MTP), for a direct apples-to-apples
# TP=2-vs-PP=2 comparison. The J.18 first PP=2 run was EAGER + no-MTP (decode floor ~6 t/s); to compare against
# the production TP=2 push-ar path (GRAPH=1 + MTP spec=3, 30 t/s) PP=2 needs the SAME capture + MTP.
#
# It REPLICATES the rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp shelf env (GDN kernels, MTP-graft BF16 drafter shim,
# PIECEWISE capture, splitting_ops) but flips TP=2 -> PP=2/TP=1. lib.sh now understands PP (b70_multicard), so
# this goes through the guarded dispatch (pre-flight probe, graceful teardown, post-verdict) + b70_bench.
#
# Notes / open risks (P2P_GPU.md J.18):
#  - vLLM-XPU PP send/recv served coherently EAGER (J.18); whether PIECEWISE capture records the PP handoff is the
#    open question this script answers.
#  - MTP head sits on the last layers -> PP stage 1, TP=1, so the spec-verify is LOCAL (no TP all_gather). The
#    shelf's capture-safe all_gather shim is a TP artifact and should be inert here; we still mount the shim for
#    its OTHER job -- building the MTP drafter unquantized/BF16 (else it loads W8A8 -> 0% accept).
#  - If MTP destabilizes capture, re-run with MTPTOK= (empty) for PP=2 GRAPH=1 no-MTP, still a valid TP/PP compare.
#
# Usage:
#   B70_AUTO_RESET=1 ./bin/gpu-run bash scripts/110_serve_pp2_graph_mtp.sh smoke   # coherence-gated
#   B70_AUTO_RESET=1 ./bin/gpu-run bash scripts/110_serve_pp2_graph_mtp.sh run     # + bench (IN/OUT/CONC)
#   bash scripts/110_serve_pp2_graph_mtp.sh stop
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SHELF="$REPO/rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"

export IMG="${IMG:-vllm-xpu-env:int8g}"
export CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
export SERVED="${SERVED:-qwen36-27b-w8a8-pp2-graph-mtp}"   # distinct id encodes PP + graph + mtp
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-1}"                        # TP=1 within each pipeline stage
export PP="${PP:-2}"                         # 2 pipeline stages, one per card
export GRAPH="${GRAPH:-1}"                   # PIECEWISE capture (the production decode lever)
export IGP="${IGP:-false}"                   # legacy splitter (mixed W8A8(scale)+BF16-GDN region)
export NOMM="${NOMM:-1}"                     # 27B is a VLM -> text-only
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export MTPTOK="${MTPTOK-3}"                  # MTP spec tokens (set MTPTOK= empty to drop MTP if it destabilizes)
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"
export COMPILESZ="${COMPILESZ-}"            # empty for spec-decode
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SHELF/patches:/opt/mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/mtp_shim )

echo "=== PP=2 GRAPH=1 MTP serve :: SERVED=$SERVED IMG=$IMG MTPTOK=${MTPTOK:-none} ==="
source "$SHELF/../_common/lib.sh"
b70_dispatch "$@"
