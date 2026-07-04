#!/usr/bin/env bash
# Qwen3.6-35B-A3B Quark W8A8 INT8 (TRUE int8 MoE) -- 2x B70, TP=2. See ./README.md for the full recipe.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (TP=2, captured), wait healthy, gen-probe, stay up
#   bash serve.sh stop                            # stop + release the GPU
#   bash serve.sh bench                           # concurrency sweep vs the running server
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + bench + stop, PIECEWISE graph capture
#
# [!! 2026-07-04 -- PORTED to vLLM v0.24.0 (torch 2.12), IMG vllm-xpu-env:v0240. Rollback =
#     IMG=vllm-xpu-env:v0230 + PATCH=patches/quark.py (the old file is kept for exactly that).
#     The v0.24.0 patch is patches/quark_v0240.py: same proven design re-grafted onto the drifted
#     upstream file -- the int8 LINEAR layers (linear_attn.*, mlp.shared_expert.*) go to a weight-only
#     int8->bf16 dequant GEMM (stock QuarkW8A8Int8 still KeyErrors on XPU: _POSSIBLE_INT8_KERNELS has
#     no XPU entry in v0.24.0 either); the 256 routed experts stay TRUE int8 via the in-tree Triton
#     fused_moe (QuarkW8A8Int8MoEMethod, unchanged upstream). gdn_attention comes from the PACKAGED
#     vllm_xpu_kernels .so (verified present) -> no kernel mount. Optional A/B: B70_INT8_LINEAR=native
#     falls through to stock QuarkW8A8Int8 -- requires IMG=vllm-xpu-env:int8g-v0240 + the runtime
#     int8 _xpu_C .so mounts (see the dense 27b-w8a8 entry); NOT the default (old FINDINGS: int8
#     linear is no speed win on this MoE -- linear is the minority path).]
#
# NEVER llm-scaler 0.14.x (no _moe_C -> int8 MoE hard-fails). Pure-Python patch -> bind-mounted
# per-container (mount-not-bake: it cannot affect any other model's container). See ORGANIZATION.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0240}"
export CKPT="${CKPT:-/models/qwen3.6-35b-a3b/quark-w8a8-int8}"
export SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
export QUANT="${QUANT:-quark}"
export TP="${TP:-2}"                        # int8 weights ~35 GB -> 17.5 GiB/card; does NOT fit one card
export GRAPH="${GRAPH:-1}"                  # DEFAULT=1: PIECEWISE capture is 8.7x decode over eager on this MoE (Lever B)
export CGMODE="${CGMODE:-PIECEWISE}"
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.92}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-8}"
export CAPSIZES="${CAPSIZES:-1,2,4,8}"
export NOMM="${NOMM-}"                      # vision ON by default on v0.24.0 (AOT fix below); NOMM=1 = text-only
export TOOLCALL="${TOOLCALL:-1}"
export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"

# Hybrid-GDN splitting_ops (same list the dense 27B v0.24.0 entry validated).
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

# IROP -- REQUIRED for capture. ROOT-CAUSE fix for the IGC/ocloc compile crash on this MoE:
# vLLM lowers rms_norm/fused_add_rms_norm to ["native"] (decomposed primitives) on XPU-with-inductor,
# so inductor fuses the RMSNorm reduction into the small-N MoE router/proj matmul -> a fused reduction
# kernel IGC cannot compile in ANY GRF mode ("Floating point exception", ocloc err 245). Forcing the
# opaque "xpu_kernels" custom-op impl makes rms_norm unfusable -> no oversized kernel. This is cleaner
# and more complete than the inductor prologue/epilogue-fusion knobs (which only caught the int4
# variant): the int8 MoE produced a DIFFERENT mm+rms_norm fusion (triton_red_fused__to_copy_add_mm_
# rms_norm_t) that survived prologue+epilogue OFF -- opaque rms_norm kills both. Empty = crash.
export IROP="${IROP:-{\"rms_norm\":[\"xpu_kernels\",\"native\"],\"fused_add_rms_norm\":[\"xpu_kernels\",\"native\"]}}"
# The int8 MoE needs BOTH levers: IROP alone left a GDN-region residual+mm+rms_norm fusion crashing
# IGC; disabling inductor prologue/epilogue/combo fusion closes that too. (int4 needed only one of
# these; int8 is harder -- its dequant-linear + true-int8-expert graph produces more reduction+mm
# fusions.) Both together = no IGC-uncompilable fused kernel. Empty = crash at capture.
export INDUCTOR="${INDUCTOR:-{\"combo_kernels\":false,\"benchmark_combo_kernel\":false,\"prologue_fusion\":false,\"epilogue_fusion\":false}}"

# Vision + torch.compile on v0.24.0 XPU needs VLLM_USE_AOT_COMPILE=0 (dense-entry fix; env-only).
DOCKER_ENV=()
if [ -z "$NOMM" ] && [ "${GRAPH:-1}" = 1 ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--skip-mm-profiling"
  DOCKER_ENV+=( -e VLLM_USE_AOT_COMPILE=0 )
fi

# THE ONE PATCH: pick the file matching the image generation (v0240 default; quark.py = v0230 rollback).
case "$IMG" in
  *v0240*) PATCH="$SCRIPT_DIR/patches/quark_v0240.py" ;;
  *)       PATCH="$SCRIPT_DIR/patches/quark.py" ;;
esac
[ -f "$PATCH" ] || { echo "[!] missing patch: $PATCH"; exit 2; }
Q1=/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark.py
Q2=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py
# NOTE: arrays cannot be `export`ed in bash; serve.sh sources lib.sh (same shell) so a plain array is visible.
MOUNTS=( -v "$PATCH:$Q1:ro" -v "$PATCH:$Q2:ro" )

# B70_INT8_LINEAR=native A/B plumbing: stock int8 dispatch needs the oneDNN kernel registry (int8g
# bake) + the runtime .so. Refuse the misconfiguration instead of crashing mid-load.
if [ "${B70_INT8_LINEAR:-dequant}" != "dequant" ]; then
  case "$IMG" in *int8g*) : ;; *) echo "[!] B70_INT8_LINEAR=native needs IMG=vllm-xpu-env:int8g-v0240"; exit 2 ;; esac
  ROOT="${ROOT:-/mnt/vm_8tb/b70}"
  PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
  MOUNTS+=( -v "$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so:$PKGD/_xpu_C.abi3.so:ro"
            -v "$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  DOCKER_ENV+=( -e B70_INT8_LINEAR="$B70_INT8_LINEAR" )
fi

# B70_EXTRA_ENV: space-separated NAME=VAL list injected as -e flags (test knobs without recipe edits).
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV} ===" >&2
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
