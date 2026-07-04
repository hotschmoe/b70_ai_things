#!/usr/bin/env bash
# serve_nvfp4_moe_35b.sh -- nvidia/Qwen3.6-35B-A3B-NVFP4 (ModelOpt MIXED_PRECISION MoE:
# 256 routed experts W4A16_NVFP4 g16 + FP8 attention/GDN + FP8 KV + bf16 vision/router/mtp).
# BRING-UP scaffold (Track 11f). Arch = Qwen3_5MoeForConditionalGeneration (GDN hybrid VLM MoE).
#
# The routed experts are per-expert f4_e2m1 tensors (experts.N.{gate,up,down}_proj.weight),
# NOT int4/int8-packed -> vLLM's cutlass/marlin/flashinfer NvFp4 MoE backends are all CUDA-only.
# The XPU bring-up path uses the EMULATION MoE backend (--moe-backend emulation): it dequantizes
# each active expert's NVFP4 weights on the fly to BF16 and runs the stock TritonExperts (which
# already works on XPU for the int4/w8a8 MoE). Slow but coherent -- the feasibility reference.
# The NVFP4 linear layers (shared_expert, attn projections that are W4A16) ride the same XPU shim
# as the 27B (patches/sitecustomize.py). FP8 attn layers ride vLLM's XPUFP8ScaledMMLinearKernel.
#
#   CARD=0 bash serve_nvfp4_moe_35b.sh          # eager single-card bring-up (container nvfp4_moe_35b)
#   bash serve_nvfp4_moe_35b.sh stop
# Run under: gpu-run --card 0 bash serve_nvfp4_moe_35b.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
DIR="$REPO/vllm/nvfp4"
SHIMDIR="${SHIMDIR:-$DIR/patches}"   # override to mount a worktree copy of the shim

IMG="${IMG:-vllm-xpu-env:int8g-v0240}"
NAME="${NAME:-nvfp4_moe_35b}"
PORT="${PORT:-8081}"
CARD="${CARD:-0}"
TP="${TP:-1}"
MODE="${MODE:-emul}"                 # emul = pure-emulation NVFP4 linear (safest for bring-up); fused = XPU kernel
MOEBACKEND="${MOEBACKEND:-emulation}"  # NVFP4 routed-expert path: emulation dequant->TritonExperts (XPU-clean)
# MOE routed-expert compute: emul = dequant-all-256-experts-to-bf16/forward (slow, correctness ref);
# fused = per-active-expert nvfp4_gemm_w4a16 (weights stay 4-bit resident, shim block 7). Default:
# fused whenever MODE=fused (the fused .so with the nvfp4 op is mounted then), else emul.
if [ "$MODE" = fused ]; then MOEFUSED="${MOEFUSED:-1}"; else MOEFUSED="${MOEFUSED:-0}"; fi
MAXLEN="${MAXLEN:-8192}"
UTIL="${UTIL:-0.90}"
MAXSEQS="${MAXSEQS:-8}"
GRAPH="${GRAPH:-0}"                   # eager first: the MoE hits an IGC capture crash needing INDUCTOR/IROP knobs
CGMODE="${CGMODE:-PIECEWISE}"
IGP="${IGP:-false}"
CAPSIZES="${CAPSIZES:-}"
MTPTOK="${MTPTOK:-}"
CKPT="${CKPT:-/models/qwen3.6-35b-a3b/nvfp4-modelopt}"
SERVED="qwen3.6-35b-a3b-NVFP4-modelopt-${MODE}-moe${MOEBACKEND}"

if [ "${1:-}" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
GRAPH_ARGS=( --enforce-eager )
GRAPH_ENV=( )
if [ "$GRAPH" = 1 ]; then
  CAP=""; [ -n "$CAPSIZES" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  SPLIT="\"splitting_ops\":[${SPLITOPS:-$_ATTN_OPS}],"
  # MoE capture needs INDUCTOR fusion-off (IGC crash on fused rms_norm+router-mm) + the MLA-pass disable.
  PASSCFG="\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":false},"
  IND="\"inductor_compile_config\":{\"combo_kernels\":false,\"benchmark_combo_kernel\":false,\"prologue_fusion\":false},"
  GRAPH_ARGS=( --compilation-config "{${CAP}${SPLIT}${PASSCFG}${IND}\"cudagraph_mode\":\"$CGMODE\",\"use_inductor_graph_partition\":$IGP}" )
  GRAPH_ENV=( -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_USE_AOT_COMPILE=0 )
  SERVED="${SERVED}-graph"
fi

SPEC_ARGS=( )
if [ -n "$MTPTOK" ]; then
  SPEC_ARGS=( --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTPTOK}}" )
  SERVED="${SERVED}-mtp${MTPTOK}"
fi

PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/w8a8_kernel_v0240/libgdn_attn_kernels_xe_2.so}"
[ -f "$GDN_SO" ] || { echo "MISSING GDN .so $GDN_SO"; exit 1; }
KERN_MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
if [ "$MODE" = fused ]; then
  FUSED_SO="${FUSED_SO:-$ROOT/nvfp4_fused_kernel_gdn/_xpu_C.abi3.so}"
  [ -f "$FUSED_SO" ] || { echo "MISSING fused GDN kernel $FUSED_SO"; exit 1; }
  KERN_MOUNTS=( -v "$FUSED_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

TP_ARGS=( )
if [ "$TP" = 1 ]; then
  MGPU=( -e ZE_AFFINITY_MASK="$CARD" ); SHM=16g
else
  SK=$([ "$GRAPH" = 1 ] && echo 1 || echo 0)
  MGPU=( -e CCL_ENABLE_SYCL_KERNELS="$SK" -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
         -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi
         -e CCL_TOPO_P2P_ACCESS="${P2PACCESS:-0}" -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}" )
  SHM=32g
  TP_ARGS=( -tp "$TP" )
fi

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$SHIMDIR:/opt/nvfp4_shim:ro" \
  "${KERN_MOUNTS[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONPATH=/opt/nvfp4_shim -e NVFP4_XPU_MODE="$MODE" -e NVFP4_MOE_W4A16_EMUL=1 \
  -e NVFP4_MOE_FUSED="$MOEFUSED" \
  "${MGPU[@]}" "${GRAPH_ENV[@]}" \
  --entrypoint vllm "$IMG" \
  serve "$CKPT" --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" --moe-backend "$MOEBACKEND" \
  --max-num-batched-tokens "${MAXNUMBATCHED:-2048}" \
  "${TP_ARGS[@]}" "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" --no-enable-prefix-caching --trust-remote-code --skip-mm-profiling

echo "container $NAME up (port $PORT, moe-backend=$MOEBACKEND, mode=$MODE, graph=$GRAPH); logs: docker logs -f $NAME"
