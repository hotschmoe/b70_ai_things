#!/usr/bin/env bash
# Qwen3.6-27B W4A8 (int4 weights / int8 activations, SmoothQuant+GPTQ, prepacked offline;
# GDN + lm_head + visual + mtp kept bf16 in-checkpoint). Single card (default card 1, the research
# card), vLLM 0.25.1, PIECEWISE capture + MTP spec=3, vision ON, fp16 KV.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   NAME=w4a8_c1 PORT=18079 ./bin/gpu-run --card 1 bash serve.sh start
#   bash serve.sh stop
#
# [!! 2026-07-21 -- PORTED TO v0.25.1 (UNVERIFIED until the coordinator gate runs it): IMG ->
#     int8g-v0251, TP=1 captured+MTP. KEY DRIFT FOUND DURING THE PORT: 0.25.1 UPSTREAMED our two
#     v0.23 W4A8 classes -- the image's mixed_precision/xpu.py now SHIPS XPUwNa16LinearKernel +
#     XPUW4A8IntLinearKernel, and linear/__init__.py already lists XPUW4A8IntLinearKernel FIRST in
#     _POSSIBLE_KERNELS[PlatformEnum.XPU] (both verified in-image 2026-07-21). So NO kernel-registry
#     mount is needed anymore; the two patch mounts below carry only the b70 deltas (prepacked-load
#     + optional hybrid w4a16 small-M route) on top of verbatim upstream file content.
#     Rollback: git history of this dir (v0.23 recipe, IMG=vllm-xpu-env:int8g).]
#
# THREE things beyond a plain single-card W4A8 serve, all wired below:
#  (1) PREPACK loader: patches/compressed_tensors_w4a8_int.py (int32 [out, in/8] weight alloc) +
#      patches/xpu.py (skip the on-load pack) + VLLM_W4A8_PREPACKED=1. The checkpoint is prepacked
#      offline (quantization_config.is_prepacked_w4a8: true); loading it unpacked needs a ~28 GiB
#      GPU transient that hangs/OOMs a 32 GB B70 on the 27B.
#  (2) GDN: Qwen3.6-27B is gated-delta-net; the :int8g image's baked kernel package ships GDN OFF ->
#      mount the GDN-enabled torch-2.12 _xpu_C.abi3.so (+ sibling libgdn) from $ROOT/w8a8_kernel_v0240
#      (the SAME .so the W8A8 v0251 shelf mounts; it also carries int4_gemm_w4a8 + int4_gemm_w4a16,
#      ABI-verified -- no separate int4 kernel build).
#  (3) MTP graft: the checkpoint carries 15 BF16 mtp.* tensors; patches/sitecustomize.py block (1)
#      builds ONLY the drafter unquantized (else it loads through the W4A8 quant path -> 0% accept).
#      Block (7) is the NEO graph-replay reclaim (single-card captured+MTP is NOT exempt from the
#      linear_stream.h:84 replay-accumulation abort -- it is transport-agnostic, 2026-07-08).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"     # needed to reference the host kernel .so before sourcing lib.sh

export IMG="${IMG:-vllm-xpu-env:int8g-v0251}"
export CKPT="${CKPT:-/models/qwen3.6-27b/w4a8-sqgptq}"
export SERVED="${SERVED:-qwen36-27b-w4a8-sqgptq}"
export DTYPE="${DTYPE:-float16}"            # int4_gemm_w4a8/w4a16 emit fp16 (hard-coded in C++); fp16
                                            # model dtype avoids a per-linear fp16->bf16 cast AND the
                                            # kernel's warning_once. If fp16 GDN/attention numerics ever
                                            # misbehave, DTYPE=auto (bf16) is the rollback -- the kernel
                                            # casts its output back to the model dtype either way.
export TP="${TP:-1}"                        # fits ONE card (int4 weights ~15 GiB); CARD picks which.
export DEVICE="${DEVICE:-${CARD:-1}}"       # card pin (lib.sh -> ZE_AFFINITY_MASK). Default 1 = the
                                            # research card (card 0 usually holds the daily driver).
                                            # Pair with: ./bin/gpu-run --card $DEVICE
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture, the decode lever
export CGMODE="${CGMODE:-PIECEWISE}"
export IGP="${IGP:-false}"                  # legacy piecewise splitter: this hybrid mixes W4A8-quantized
                                            # (weight_scale) + unquantized BF16-GDN linears in one captured
                                            # region; the inductor partitioner KeyErrors on exactly that mix
                                            # (W8A8 shelf, same arch). Defensive default; IGP=true to A/B.
export NOMM="${NOMM-}"                       # VISION ON by default (checkpoint keeps all 333 bf16
                                            # model.visual.* tensors). NOMM=1 for a text-only perf bench.
export KVDTYPE="${KVDTYPE:-}"               # fp16 KV. Do NOT set fp8 here: uncalibrated fp8 KV is the
                                            # proven repetition/garbage trap on this family (NVFP4 DD
                                            # root-cause, 2026-07-08) and sglang-XPU rejects it outright.
export UTIL="${UTIL:-0.85}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-4}"
export MTPTOK="${MTPTOK:-3}"                # lib.sh builds --speculative-config {"method":"mtp",...}.
                                            # spec=3 mirrors the W8A8 27B sweep winner (scripts/111:
                                            # monotonic decreasing past 3); not yet re-swept for W4A8.
export CAPSIZES="${CAPSIZES:-1,2,4}"        # covers the c1 MTP verify batch (1+spec=4). Multi-seq decode
                                            # (up to MAXSEQS*4=16 tokens) falls back to eager -- fine for
                                            # the single-stream research target; widen if benching c4.
export COMPILESZ="${COMPILESZ-}"            # MUST be empty for spec-decode (compile_sizes [1] is rejected)
# SPLITOPS: left EMPTY on purpose -> vLLM's default attention/GDN splitting_ops. The W8A8 shelf's
# explicit _ATTN_OPS list exists for its TP=2+MTP collective history; v0.25.1 asserts splitting_ops is
# a SUPERSET of CompilationConfig._attention_ops only when you SET it, so empty is the safe default here.

# VISION + capture on v0.24.0+ XPU: standalone AOT compile mishandles the optional/None multimodal
# inputs ('NoneType' has no attribute 'size' at init). Fix = VLLM_USE_AOT_COMPILE=0 (no runtime perf
# cost; AOT is only startup serialization) + --skip-mm-profiling. Same auto-gate as the W8A8 shelf.
if [ -z "$NOMM" ] && [ "${GRAPH:-1}" = 1 ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--skip-mm-profiling"
  export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }VLLM_USE_AOT_COMPILE=0"
fi

KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
# torch-2.12 combined int8+int4+GDN .so, shared with the W8A8 v0251 shelf (ABI-verified there).
GDN_SO="${GDN_SO:-$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$SCRIPT_DIR/patches/xpu.py:$KP:ro"
         -v "$SCRIPT_DIR/patches/compressed_tensors_w4a8_int.py:$SP:ro"
         -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro"
         -v "$SCRIPT_DIR/patches:/opt/mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/mtp_shim
             -e VLLM_W4A8_PREPACKED=1
             -e B70_W4A8_HYBRID="${B70_W4A8_HYBRID:-0}" )
# B70_W4A8_HYBRID (default 0=OFF): route M<=N linears through the quant-free fp16-act int4_gemm_w4a16
# op (patches/xpu.py delta 2; sglang-measured 1.83x at M==1). Set 1 for decode-only, 4 to also cover
# the MTP spec=3 verify batch. Zero extra weight memory (shares the packed-weight NT view) -- unlike
# the W8A8 W8A16 route this does NOT duplicate the weight layout. UNMEASURED on vLLM: gate before
# defaulting on.
#
# [!] env-append ORDERING HYGIENE (the W8A8 push-AR lesson, commit cd8848a): DOCKER_ENV is ASSIGNED
# exactly once, above. Every block below must APPEND with DOCKER_ENV+=( ... ) -- never reassign --
# or earlier -e flags are silently dropped. This serve has no push-AR block (TP=1, no collectives),
# but keep the discipline if one is ever added.

# NEO graph-replay leak RECLAIM -- DEFAULT ON for GRAPH=1: without it a captured+MTP serve hits the
# NEO linear_stream.h:84 replay-accumulation abort (transport-agnostic -- single-card is NOT exempt;
# root cause + fix 2026-07-08, docs/20260707_dd_mtp_piecewise_neo_abort.md). Implemented in
# patches/sitecustomize.py block (7): keep_graph=True + re-instantiate each captured XPUGraph every N
# replays, zero throughput cost. Opt out CGRECLAIM=0 (then B70_XPU_DRAFTER_EAGER=1 is the fallback
# leak fix, block (6)). No-op when GRAPH=0.
if [ "${GRAPH:-1}" = 1 ] && [ "${CGRECLAIM:-1000}" != 0 ] && [[ "${B70_EXTRA_ENV:-}" != *B70_XPU_CG_RECLAIM* ]]; then
  DOCKER_ENV+=( -e "B70_XPU_CG_RECLAIM=${CGRECLAIM:-1000}" )
  echo "=== NEO-leak reclaim ON: B70_XPU_CG_RECLAIM=${CGRECLAIM:-1000} (re-instantiate captured graphs) ===" >&2
fi

# --- DIAGNOSTIC / BISECT TOGGLES (opt-in; default-off => the serve command is byte-identical) -------
#   B70_NOMTP=1  drop --speculative-config (MTP OFF) -- the bisect variable.
#   B70_DEBUG=1  faulthandler: dump a worker py+C traceback on a FATAL signal.
if [ "${B70_NOMTP:-0}" = 1 ]; then export MTPTOK=""; export SPEC=""; echo "=== B70_NOMTP=1 -> MTP OFF (no --speculative-config) ===" >&2; fi
if [ "${B70_DEBUG:-0}" != 0 ]; then
  DOCKER_ENV+=( -e PYTHONFAULTHANDLER=1 -e PYTHONUNBUFFERED=1 )
  echo "=== B70_DEBUG=${B70_DEBUG} -> faulthandler injected ===" >&2
fi
# B70_EXTRA_ENV: space-separated NAME=VAL list injected as -e flags (test any env without recipe edits).
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV} ===" >&2
fi
# B70_EXTRA_MOUNTS: space-separated host:container[:ro] bind specs appended to MOUNTS.
if [ -n "${B70_EXTRA_MOUNTS:-}" ]; then
  for m in ${B70_EXTRA_MOUNTS}; do MOUNTS+=( -v "$m" ); done
  echo "=== B70_EXTRA_MOUNTS -> injected: ${B70_EXTRA_MOUNTS} ===" >&2
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
