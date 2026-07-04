#!/usr/bin/env bash
# Qwen3.6-35B-A3B MoE int4-AutoRound (W4A16 experts) -- FASTEST single-card decode. PIECEWISE captured.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh      # start (GRAPH capture), wait healthy, gen-probe, stay up
#   bash serve.sh stop                                   # stop + release the GPU
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run   # serve + bench + stop in one lease
#
# [!! 2026-07-04 -- PORTED to vLLM v0.24.0 (torch 2.12), IMG vllm-xpu-env:v0240. Rollback =
#     IMG=vllm-xpu-env:v0230moe (the old BAKED-patch leaf; the patch mounts below are harmless there
#     but wrong-version -- drop them if rolling back).
#     v0.24.0 rewrote INC into a package (inc/ + schemes/); the old inc.py RoutedExperts->MoeWNA16
#     routing patch (contrib/vllm_moe_xpu) is re-ported as patches/inc_wna16_scheme.py, now
#     MOUNT-not-bake: upstream get_moe_method still hard-returns UnquantizedFusedMoEMethod on XPU
#     (bf16-inflates the 256 int4 experts -> ~70 GB OOM); the patch routes gptq/awq-packed experts
#     to the pure-Triton MoeWNA16 path, skipping the CUDA-only Marlin probes. Linear int4 layers now
#     ride v0.24.0's IN-TREE INC XPU path (torch.ops._xpu_C.int4_gemm_w4a16, packaged .so -- which
#     also packages gdn_attention, so NO kernel .so mount is needed at all).]
#
# Old v0230moe numbers: decode ~56.8 t/s captured (fp16 KV) / ~67.7 t/s with fp8 KV. ~21 GiB model.
# Fits ONE 32 GB B70. Aggregate plateaus ~206 t/s at N>=8 (routed-expert union -> all 256 experts).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0240}"
export CKPT="${CKPT:-/models/qwen3.6-35b-a3b/int4-autoround}"
export SERVED="${SERVED:-qwen36-35b-a3b-int4}"
export GRAPH="${GRAPH:-1}"
export CGMODE="${CGMODE:-PIECEWISE}"
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-64}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32,64}"
export KVDTYPE="${KVDTYPE:-fp8_e5m2}"      # fp8-storage KV -> ~65 t/s + 2x ctx/batch (B70 has no FP8 ALU)
export TOOLCALL="${TOOLCALL:-1}"
export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"
export NOMM="${NOMM-}"                      # vision ON by default on v0.24.0 (AOT fix below); NOMM=1 = text-only

# Hybrid-GDN splitting_ops (same list the dense 27B v0.24.0 entry validated): the attention + GDN
# custom ops are the genuine non-capturable ops and must partition the PIECEWISE graph.
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
export SPLITOPS="${SPLITOPS:-${_ATTN_OPS}}"

# INDUCTOR fusion opt-out -- REQUIRED for capture on this MoE (v0.24.0 + Battlemage IGC).
# Symptom: PIECEWISE capture aborts compiling triton_red_fused__to_copy_add_fused_add_rms_norm_mm_t
# with "IGC: Internal Compiler Error: Floating point exception" (ocloc err 245) -- and it fails in
# BOTH default AND 256-GRF mode (Intel triton retries large-GRF, that crashes too), so the SPIR-V is
# uncompilable by IGC in any register mode. ROOT CAUSE: inductor prologue/epilogue-fuses the RMSNorm
# reduction into the small-N MoE router-gate matmul (hidden 2048 -> 256 experts) -> an oversized
# fused reduction+mm kernel IGC chokes on. The dense 27B never hits it (its matmuls are large-N ->
# DPAS templates, no reduction fusion). FIX = disable prologue+epilogue fusion (pure optimizations,
# correctness identical) so the norm reduction and the router mm compile as separate small kernels.
# combo_kernels also off (v0.24.0 force-on; unneeded here). Empty INDUCTOR = vLLM default (WILL crash).
export INDUCTOR="${INDUCTOR:-{\"combo_kernels\":false,\"benchmark_combo_kernel\":false,\"prologue_fusion\":false}}"

# Vision + torch.compile on v0.24.0 XPU: the standalone AOT compile mishandles the optional/None
# multimodal inputs ('NoneType'.size dynamo crash) -> VLLM_USE_AOT_COMPILE=0 (env-only fix, no
# runtime cost; same as the dense 27B entry). Applied when vision is on AND capture is on.
DOCKER_ENV=()
if [ -z "$NOMM" ] && [ "${GRAPH:-1}" = 1 ]; then
  export EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--skip-mm-profiling"
  DOCKER_ENV+=( -e VLLM_USE_AOT_COMPILE=0 )
fi

# THE ONE PATCH (mount-not-bake): inc_wna16_scheme.py over both resolvable vllm locations
# (/workspace editable install is the one Python actually imports; venv site-packages for safety).
PATCH="$SCRIPT_DIR/patches/inc_wna16_scheme.py"
[ -f "$PATCH" ] || { echo "[!] missing patch: $PATCH"; exit 2; }
P1=/workspace/vllm/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py
P2=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py
MOUNTS=( -v "$PATCH:$P1:ro" -v "$PATCH:$P2:ro" )

# B70_EXTRA_ENV: space-separated NAME=VAL list injected as -e flags (test knobs without recipe edits).
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV} ===" >&2
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
