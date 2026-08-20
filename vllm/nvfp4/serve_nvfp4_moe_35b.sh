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
SERVED="${SERVED:-qwen3.6-35b-a3b-NVFP4-modelopt-${MODE}-moe${MOEBACKEND}}"

if [ "${1:-}" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

# v0.26 PIECEWISE requires VLLM_COMPILE. Forcing inductor_compile_config +
# splitting_ops on TP=1 (the old 35B bring-up knobs) sets compilation_mode=3
# and engine-init asserts (L59 G1g/T2g BOOTFAIL). Match the 27B recipe:
# TP=1 = no split list; TP>1 = attn/GDN split + MLA pass off.
# hpc_rope_norm_forward is in v0.25.1+ _attention_ops and must be listed
# whenever we do pass a split list.
_ATTN_OPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward"'
GRAPH_ARGS=( --enforce-eager )
GRAPH_ENV=( )
if [ "$GRAPH" = 1 ]; then
  if [ "$TP" != 1 ] && [ -n "${MTPTOK:-}" ] && [ -n "$CAPSIZES" ]; then
    _MAXCAP=$(echo "$CAPSIZES" | tr ',' '\n' | sort -n | tail -1)
    if [ "$MAXSEQS" -lt "$_MAXCAP" ] 2>/dev/null; then
      echo "[guard] TP>1+MTP+capture: raising MAXSEQS $MAXSEQS -> $_MAXCAP" >&2
      MAXSEQS="$_MAXCAP"
    fi
  fi
  CAP=""; [ -n "$CAPSIZES" ] && CAP="\"cudagraph_capture_sizes\":[$CAPSIZES],"
  SPLIT=""; PASSCFG=""; IND=""
  if [ "$TP" != 1 ]; then
    SPLIT="\"splitting_ops\":[${SPLITOPS:-$_ATTN_OPS}],"
    PASSCFG="\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":false},"
  fi
  [ -n "${INDUCTOR:-}" ] && IND="\"inductor_compile_config\":${INDUCTOR},"
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

# KV_FP8=0: strip checkpoint kv_cache_scheme so vLLM does not force uncalibrated
# fp8_e4m3 KV (D18 !!!! / 27B Track 11h). Use --kv-cache-dtype auto (model
# dtype). Do NOT pass bfloat16 -- flash_attn XPU rejects that enum.
KV_MOUNTS=( )
if [ "${KV_FP8:-1}" = 0 ]; then
  _HOST_CKPT="${HOST_CKPT:-}"
  if [ -z "$_HOST_CKPT" ]; then
    _rel="${CKPT#/models/}"
    _HOST_CKPT="$REPO/models/files/$_rel"
  fi
  _CFG_SRC="$_HOST_CKPT/config.json"
  _CFG_PATCH="${KV_PATCH_DIR:-/tmp}/b70_ornith_nokvfp8.json"
  python3 -c "import json; d=json.load(open('$_CFG_SRC')); d.get('quantization_config',{}).pop('kv_cache_scheme',None); json.dump(d,open('$_CFG_PATCH','w'))" \
    || { echo "failed to generate $_CFG_PATCH from $_CFG_SRC"; exit 1; }
  KV_MOUNTS=( -v "$_CFG_PATCH:$CKPT/config.json:ro" )
  KVDTYPE="${KVDTYPE:-auto}"
  echo "=== KV_FP8=0: kv_cache_scheme stripped, KVDTYPE=$KVDTYPE ==="
fi

LANG_ARGS=( )
if [ "${LANGONLY:-0}" = 1 ]; then
  LANG_ARGS=( --language-model-only )
  echo "=== LANGONLY=1 --language-model-only ==="
fi

# push-AR (TP>1 only). Same wiring as serve_nvfp4_27b.sh. Ornith hidden=2048
# so decode numel is 8*2048=16384; PUSH_AR_GRAPH=1 uses MIN_NUMEL=0.
PUSH_AR="${PUSH_AR:-0}"
PUSH_AR_MOUNTS=( ); PUSH_AR_ENV=( )
if [ "$PUSH_AR" = 1 ] && [ "$TP" != 1 ]; then
  PUSH_AR_DIR="$REPO/vllm/contrib/vllm_push_allreduce"
  PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-0}"
  if [ "$PUSH_AR_GRAPH" = 1 ] && { [ "$GRAPH" != 1 ] || [ "$CGMODE" = NONE ]; }; then
    echo "[guard] PUSH_AR_GRAPH=1 needs GRAPH=1; forcing 0" >&2
    PUSH_AR_GRAPH=0
  fi
  if [ "$PUSH_AR_GRAPH" = 1 ]; then
    _PA_SO_NAME="libxpu_push_ar_graph.so"; _PA_MIN_DEF=0
  else
    _PA_SO_NAME="libxpu_push_ar_torch.so"; _PA_MIN_DEF=65536
  fi
  PUSH_AR_SO_HOST="$PUSH_AR_DIR/prebuilt/$_PA_SO_NAME"
  [ -f "$PUSH_AR_SO_HOST" ] || { echo "MISSING push-ar .so $PUSH_AR_SO_HOST"; exit 1; }
  PUSH_AR_MOUNTS=( -v "$PUSH_AR_DIR:/opt/push_ar:ro" )
  PUSH_AR_ENV=( -e PUSH_AR_PATCH=/opt/push_ar/_push_ar_patch.py
                -e PUSH_AR_SO="/opt/push_ar/prebuilt/$_PA_SO_NAME"
                -e PUSH_AR_DISABLE=0
                -e PUSH_AR_GRAPH="$PUSH_AR_GRAPH"
                -e PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-$_PA_MIN_DEF}"
                -e PUSH_AR_MAXB="${PUSH_AR_MAXB:-134217728}" )
  echo "=== PUSH_AR ON graph=$PUSH_AR_GRAPH so=$_PA_SO_NAME ==="
  SERVED="${SERVED}-pushar"
fi

# O4c sidecar NVFP4 M=1 GEMV .so. Optional. Do not swap live _xpu_C.
M1_MOUNTS=( )
if [ -n "${B70_NVFP4_M1_SO_HOST:-}" ]; then
  [ -f "$B70_NVFP4_M1_SO_HOST" ] || { echo "MISSING M1_SO $B70_NVFP4_M1_SO_HOST"; exit 1; }
  M1_MOUNTS=( -v "$B70_NVFP4_M1_SO_HOST:/opt/nvfp4_m1/b70_nvfp4_m1_gemv.so:ro" )
  echo "=== M1_SO $B70_NVFP4_M1_SO_HOST -> /opt/nvfp4_m1/b70_nvfp4_m1_gemv.so ==="
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
  "${KERN_MOUNTS[@]}" "${KV_MOUNTS[@]}" "${PUSH_AR_MOUNTS[@]}" "${M1_MOUNTS[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONPATH=/opt/nvfp4_shim -e NVFP4_XPU_MODE="$MODE" -e NVFP4_MOE_W4A16_EMUL=1 \
  -e NVFP4_MOE_FUSED="$MOEFUSED" \
  "${MGPU[@]}" "${GRAPH_ENV[@]}" "${PUSH_AR_ENV[@]}" \
  $( [ -n "${B70_EXTRA_ENV:-}" ] && for kv in ${B70_EXTRA_ENV}; do printf -- '-e %s ' "$kv"; done ) \
  --entrypoint vllm "$IMG" \
  serve "$CKPT" --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" --moe-backend "$MOEBACKEND" \
  --max-num-batched-tokens "${MAXNUMBATCHED:-2048}" \
  ${KVDTYPE:+--kv-cache-dtype "$KVDTYPE"} \
  "${LANG_ARGS[@]}" \
  "${TP_ARGS[@]}" "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" --no-enable-prefix-caching --trust-remote-code --skip-mm-profiling

echo "container $NAME up (port $PORT, moe-backend=$MOEBACKEND, mode=$MODE, graph=$GRAPH); logs: docker logs -f $NAME"
