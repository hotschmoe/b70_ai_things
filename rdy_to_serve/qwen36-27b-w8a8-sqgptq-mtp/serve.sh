#!/usr/bin/env bash
# Qwen3.6-27B W8A8 (compressed-tensors INT8 w x INT8 a, SmoothQuant+GPTQ) + BF16 MTP graft, TP=2 + MTP spec=5.
# THE FASTEST single-stream 27B config on the 2x B70 rig: ~63-64 tok/s decode, ~3.4x vs MTP-off (scripts/93,94).
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (both cards), wait healthy, gen-probe
#   bash serve.sh stop                              # stop + release both GPUs
#   MTPTOK=0 ... (see below) to serve MTP-OFF
#
# [!] IMAGE: vllm-xpu-env:int8g  (custom oneDNN s8s8s32 INT8 W8A8 GEMM = XPUInt8ScaledMMLinearKernel, auto-selected
#     from the compressed-tensors W8A8-int8 config; plus register_fake so XPU graph capture can trace the int8 ops).
# [!] 35 GiB int8 weights -> needs TP=2 (does NOT fit one 32 GB card). TP collectives run CPU-staged oneCCL over
#     PCIe (no Battlemage P2P) -- LATENCY-bound, which is exactly what MTP hides (1 collective round per ~6 verified
#     tokens instead of per token), so MTP is a BIGGER win at TP=2 than single-card.
#
# THREE things beyond a plain W8A8 serve, all wired below:
#  (1) GDN: Qwen3.6-27B is gated-delta-net; the :int8g baked kernel ships GDN OFF -> mount the GDN-enabled
#      _xpu_C.abi3.so (+ sibling libgdn_attn_kernels_xe_2.so) over the baked one (host kernel build).
#  (2) MTP graft: CKPT is the *-mtp-graft dir (15 BF16 mtp.* tensors added). Mount patches/sitecustomize.py on
#      PYTHONPATH so ONLY the MTP drafter is built unquantized/BF16 (else it loads through the W8A8 quant path -> 0% accept).
#  (3) splitting_ops THE FIX: TP+MTP records the spec all_gather into the SYCL graph, but oneCCL 2021.17's sched
#      allgather has no graph-recordable impl -> capture crash. Listing the 3 collectives in SPLITOPS makes inductor
#      partition at them (run EAGER) while decode stays captured. This is what overturned the old "TP=2 MTP DEAD".
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"     # needed to reference the host GDN .so before sourcing lib.sh

export IMG="${IMG:-vllm-xpu-env:int8g}"
export CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
export SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-mtp}"
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture (int8g traces the custom int8 ops)
export NOMM="${NOMM:-1}"                    # 27B is a qwen3_5 VLM -> text-only
export UTIL="${UTIL:-0.90}"                 # 17 GiB/card -> plenty of KV headroom at TP=2
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export MTPTOK="${MTPTOK:-5}"                # MTP spec tokens; spec=5 is the winner (spec=6 collapses: 1-layer head)
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"    # include the spec-verify batch 1+spec=6
export COMPILESZ="${COMPILESZ-}"           # MUST be empty for spec-decode (compile_sizes [1] is rejected; pads to 1+spec)
# THE FIX (3): collectives in splitting_ops -> eager partition boundary -> spec all_gather not recorded into the graph.
# Comma-separated quoted vllm:: op list (the model's attention ops + the 3 TP collectives); _common wraps it in [...].
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
_COLLECTIVES='"vllm::all_reduce","vllm::reduce_scatter","vllm::all_gather"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS},${_COLLECTIVES}}"

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SCRIPT_DIR/patches:/opt/mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/mtp_shim )

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
