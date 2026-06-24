#!/usr/bin/env bash
# Qwen3.6-27B W8A8 (compressed-tensors INT8 w x INT8 a, SmoothQuant+GPTQ) + BF16 MTP graft, TP=2 + MTP spec=5.
# Coherent single-stream 27B int8-activation serve. EAGER ~9 t/s MTP-on (2.3x vs ~4 t/s MTP-off), 48% accept.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
# [!!! 2026-06-23 CORRECTION -- scripts/101-106]: the OLD "63-64 t/s / 3.4x captured" headline was a FALSE POSITIVE
#   benched on GARBAGE. Two stacked bugs were found+fixed/characterized:
#   (A) ROOT CAUSE of the garbage (FIXED, config-only, no requant): the checkpoint config.json `ignore` list had 336
#       ENUMERATED leaf names with the wrong flat prefix `model.layers.N.linear_attn.*` -- they do NOT match the
#       VLM-nested keys `model.language_model.layers.N.*`, so the 48 GDN linear_attn layers (correctly stored BF16)
#       were NOT exempted -> built as W8A8, BF16 silently shape-matched int8, scale missing -> 48 recurrent garbage
#       layers -> degenerate output. The degenerate body made draft==target -> trivial ~98% accept / accept_len 5.9
#       (the bench counted tokens, never read text). FIX = regex ignore (scripts/104); weights were always good
#       (dequant cos 0.97-0.9999 vs base). Eager now COHERENT.
#   (B) CAPTURED PATH STILL BROKEN: with correct BF16 GDN, PIECEWISE capture is numerically broken on this TP=2 hybrid
#       (IGP=true -> KeyError weight_scale crash; IGP=false -> serves but garbage even WITHOUT MTP; clean-cache
#       confirmed). 14B W8A8 captures fine single-card, so it's a TP=2 + BF16-GDN-in-captured-pieces + int8 capture
#       bug. -> recipe DEFAULTS TO EAGER (GRAPH=0) for correctness. `GRAPH=1` reproduces the broken captured path.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (both cards), wait healthy, COHERENCE-GATED gen-probe
#   bash serve.sh stop                              # stop + release both GPUs
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
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture (2026-06-24: Bug B FIXED). The old garbage was caused
                                            # by EJECTING the TP collectives to eager (they are out-of-place; the
                                            # ejected output does not land at the captured-piece's capture-time
                                            # address -> stale read -> garbage). Fix = eject NOTHING (see SPLITOPS)
                                            # + a capture-safe all_gather (patches/, all-reduce-of-padded) so the
                                            # spec-verify all_gather records into the SYCL graph instead of crashing.
export IGP="${IGP:-false}"                  # legacy piecewise splitter: REQUIRED on this hybrid (the inductor
                                            # partitioner KeyErrors on the mixed W8A8(scale)+BF16-GDN(no-scale) region).
export NOMM="${NOMM:-1}"                    # 27B is a qwen3_5 VLM -> text-only
export UTIL="${UTIL:-0.90}"                 # 17 GiB/card -> plenty of KV headroom at TP=2
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-8}"
export MTPTOK="${MTPTOK:-3}"                # MTP spec tokens. spec=3 = WINNER on the fixed captured path (scripts/111
                                            # coherence-gated: spec3 34.82 t/s @51% > spec4 30.56 @37% > spec5 26.10
                                            # @26% -- MONOTONIC DECREASING; the 1-layer MTP head over-drafts past ~3.
                                            # (The old "spec=5 winner / climbing 50/57/63" was on degenerate garbage.)
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"    # covers the spec-verify batch 1+spec for spec 3/4/5 (=4/5/6)
export COMPILESZ="${COMPILESZ-}"           # MUST be empty for spec-decode (compile_sizes [1] is rejected; pads to 1+spec)
# THE FIX (3) [2026-06-24, REVISED]: eject NOTHING. splitting_ops = the model's attention + GDN custom ops ONLY
# (the genuine non-capturable ops). Do NOT add the TP collectives: ejecting them breaks the captured-piece input-
# address contract -> garbage (the old config ejected all 3 -> the ejected per-layer all_reduce corrupted decode).
# all_reduce + reduce_scatter record fine inside the graph. all_gather (which oneCCL CANNOT record) is handled by
# the capture-safe all-reduce-of-padded shim in patches/sitecustomize.py, so it too stays captured. Net: every
# collective is captured + correct -> coherent body AND a numerically-correct spec-verify (real accept), all fast.
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SCRIPT_DIR/patches:/opt/mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/mtp_shim )

# OPTIONAL push-allreduce overlay -- set PUSH_AR=1 to replace oneCCL's TP all-reduce with the custom
# L0-IPC push transport (contrib/vllm_push_allreduce). CAPTURE-GATED by default (PUSH_AR_MIN_NUMEL=65536:
# prefill-only push; the small decode all-reduces stay on oneCCL inside the SYCL graph), so it composes
# with this recipe's GRAPH=1. Push-ar's sitecustomize CHAINS the MTP shim (PUSH_AR_CHAIN_SITECUSTOMIZE)
# so both patches run. P2PACCESS stays 0 (push-ar uses its own L0-IPC peer write, NOT oneCCL's wedge-prone
# CCL_TOPO_P2P_ACCESS). Measured J.17 (P2P_GPU.md): +15-55% throughput / 2.3-2.5x TTFT vs oneCCL on this
# 27B-W8A8 TP=2 serve. Default OFF -> the proven oneCCL baseline above is byte-for-byte unchanged.
if [ "${PUSH_AR:-0}" = 1 ]; then
  PUSH_AR_DIR="$SCRIPT_DIR/../../contrib/vllm_push_allreduce"
  SO_HOST="${SO_HOST:-$PUSH_AR_DIR/prebuilt/libxpu_push_ar_torch.so}"  # self-contained; rebuild via scripts/108 REBUILD_SO=1
  [ -f "$SO_HOST" ] || { echo "[!] PUSH_AR=1 but push-ar .so missing at $SO_HOST" >&2; exit 1; }
  MOUNTS+=( -v "$PUSH_AR_DIR:/opt/push_ar:ro"
            -v "$SO_HOST:/opt/push_ar_so/libxpu_push_ar_torch.so:ro" )
  DOCKER_ENV=( -e PYTHONPATH=/opt/push_ar:/opt/mtp_shim
               -e PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py
               -e PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so
               -e PUSH_AR_DISABLE=0
               -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-65536}" )
  export P2PACCESS="${P2PACCESS:-0}"
  echo "=== PUSH_AR overlay ON (capture-gated MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-65536}, P2PACCESS=0, .so=$SO_HOST) ==="
fi

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
