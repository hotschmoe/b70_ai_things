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
export CKPT="${CKPT:-/models/qwen3.6-27b/w8a8-sqgptq}"
export SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-mtp}"
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture (2026-06-24: Bug B FIXED). The old garbage was caused
                                            # by EJECTING the TP collectives to eager (they are out-of-place; the
                                            # ejected output does not land at the captured-piece's capture-time
                                            # address -> stale read -> garbage). Fix = eject NOTHING (see SPLITOPS)
                                            # + a capture-safe all_gather (patches/, all-reduce-of-padded) so the
                                            # spec-verify all_gather records into the SYCL graph instead of crashing.
export CGMODE="${CGMODE:-NONE}"             # DEFAULT NONE (2026-06-25, campaign 120): cudagraph_mode=NONE keeps
                                            # torch.compile/inductor but SKIPS graph REPLAY -> no command-stream
                                            # accumulation -> STABLE (soaked 57k tokens, ~2.9x the crash zone) and
                                            # ~2x enforce-eager (decode 25.39 vs 12.78 t/s). CGMODE=PIECEWISE is
                                            # faster (~35 t/s) but CRASHES under sustained MTP (graph-replay
                                            # accumulation in the MTP path). docs/20260625_w8a8_27b_mtp_graph_campaign.md
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

# push-allreduce overlay -- DEFAULT ON (PUSH_AR=1, PUSH_AR_GRAPH=1): replaces oneCCL's TP all-reduce with the
# custom L0-IPC push transport (contrib/vllm_push_allreduce). With PUSH_AR_GRAPH=1 the DECODE all-reduce is also
# graph-captured (device-side L0-event sync recorded into torch's XPUGraph, P2P_GPU.md K.6/K.8), so EVERY
# all-reduce (prefill AND decode) uses the 11 GB/s posted-write fabric -- no oneCCL fallback. Push-ar's
# sitecustomize CHAINS the MTP shim (PUSH_AR_CHAIN_SITECUSTOMIZE) so both patches run. P2PACCESS stays 0
# (push-ar uses its own L0-IPC peer write, NOT oneCCL's wedge-prone CCL_TOPO_P2P_ACCESS). Measured vs oneCCL on
# this 27B-W8A8 TP=2 serve: 3.8x prefill TTFT, +80-126% agg throughput (J.21), +8-10% per-stream decode (K.8).
# OPT-OUT to the plain oneCCL baseline with PUSH_AR=0. Set PUSH_AR_GRAPH=0 for the prefill-only push (decode on
# oneCCL, J.17/J.21). The .so is prebuilt + committed; rebuild via scripts/118 (graph) or scripts/108 (eager).
if [ "${PUSH_AR:-1}" = 1 ]; then
  PUSH_AR_DIR="$SCRIPT_DIR/../../contrib/vllm_push_allreduce"
  # [!] CRITICAL (2026-06-26): PUSH_AR_GRAPH=1 records the DECODE all-reduce INTO the XPU graph. CGMODE=NONE has
  # NO graph -> the captured-decode push reads a stale/uninit buffer -> INPUT-DEPENDENT GARBAGE ("!!!!" on some
  # prompts; coherent on others -> a single gen-probe misses it). So on NONE the decode all-reduce MUST stay on
  # the host-barrier/oneCCL path: default PUSH_AR_GRAPH=0 here. (verified: NONE+PUSH_AR_GRAPH=0 -> 0/8 garbage.)
  _PAG_DEF=$([ "${CGMODE:-PIECEWISE}" = NONE ] && echo 0 || echo 1)
  PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-$_PAG_DEF}"
  # PUSH_AR_GRAPH=1 -> CAPTURABLE decode path: graph .so + MIN_NUMEL=0 (push ALL all-reduces incl. decode; needs
  # a captured graph). PUSH_AR_GRAPH=0 -> prefill-only push (host-barrier .so, decode-sized -> oneCCL fallback).
  if [ "$PUSH_AR_GRAPH" = 1 ]; then
    _DEF_SO="$PUSH_AR_DIR/prebuilt/libxpu_push_ar_graph.so"; _DEF_MIN=0
  else
    _DEF_SO="$PUSH_AR_DIR/prebuilt/libxpu_push_ar_torch.so"; _DEF_MIN=65536
  fi
  SO_HOST="${SO_HOST:-$_DEF_SO}"  # self-contained; rebuild via scripts/108 REBUILD_SO=1 or scripts/118
  [ -f "$SO_HOST" ] || { echo "[!] PUSH_AR=1 but push-ar .so missing at $SO_HOST (set PUSH_AR=0 for oneCCL)" >&2; exit 1; }
  MOUNTS+=( -v "$PUSH_AR_DIR:/opt/push_ar:ro"
            -v "$SO_HOST:/opt/push_ar_so/libxpu_push_ar_torch.so:ro" )
  DOCKER_ENV=( -e PYTHONPATH=/opt/push_ar:/opt/mtp_shim
               -e PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/mtp_shim/sitecustomize.py
               -e PUSH_AR_SO=/opt/push_ar_so/libxpu_push_ar_torch.so
               -e PUSH_AR_DISABLE=0
               -e PUSH_AR_GRAPH="$PUSH_AR_GRAPH"
               -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-$_DEF_MIN}" )
  export P2PACCESS="${P2PACCESS:-0}"
  echo "=== PUSH_AR overlay ON [default] (GRAPH=${PUSH_AR_GRAPH:-1} MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-$_DEF_MIN}, P2PACCESS=0, .so=$SO_HOST) ==="
else
  echo "=== PUSH_AR overlay OFF (PUSH_AR=0 -> plain oneCCL TP all-reduce baseline) ==="
fi

# --- DIAGNOSTIC / BISECT TOGGLES (opt-in; default-off => the serve command is byte-identical) -------
# Added 2026-06-25 to root-cause the TP=2 long-load engine crash (JOURNAL/FINDINGS 2026-06-25).
#   B70_NOMTP=1  drop --speculative-config (run with MTP OFF) -- the bisect variable.
#   B70_DEBUG=1  faulthandler only: near-zero overhead, dumps a worker py+C traceback on a FATAL signal
#                (SIGSEGV/SIGABRT/SIGBUS/SIGFPE) -- the worker dies with no traceback today, this catches it.
#   B70_DEBUG=2  + Level-Zero validation layer + UR/oneCCL/vLLM debug logging (HEAVIER, perturbs timing;
#                use to characterize a crash, NOT for a clean timing bisect).
if [ "${B70_NOMTP:-0}" = 1 ]; then export MTPTOK=""; export SPEC=""; echo "=== B70_NOMTP=1 -> MTP OFF (no --speculative-config) ===" >&2; fi
if [ "${B70_DEBUG:-0}" != 0 ]; then
  DOCKER_ENV+=( -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 )
  # B70_DEBUG=2 adds the Level-Zero validation layer + basic LEAK CHECKER (prints per-handle create-vs-
  # destroy counts at exit -> confirms WHICH handle type accumulates) + UR/oneCCL/vLLM debug logging.
  [ "${B70_DEBUG}" = 2 ] && DOCKER_ENV+=( -e VLLM_LOGGING_LEVEL=DEBUG -e ZE_ENABLE_VALIDATION_LAYER=1 \
        -e ZE_ENABLE_PARAMETER_VALIDATION=1 -e ZEL_ENABLE_BASIC_LEAK_CHECKER=1 -e UR_L0_LEAKS_DEBUG=1 \
        -e UR_LOG_LEVEL=info -e CCL_LOG_LEVEL=info )
  echo "=== B70_DEBUG=${B70_DEBUG} -> diagnostic env injected (faulthandler$([ "${B70_DEBUG}" = 2 ] && echo ' + L0 validation/leak-checker + UR/CCL/vLLM debug')) ===" >&2
fi
# B70_EXTRA_ENV: space-separated NAME=VAL list injected as -e flags (test any env -- UR_L0_* event knobs,
# CCL_ENABLE_SYCL_KERNELS, etc. -- without further recipe edits). e.g. B70_EXTRA_ENV="UR_L0_REUSE_DISCARDED_EVENTS=1".
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV} ===" >&2
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
