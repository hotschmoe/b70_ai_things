#!/usr/bin/env bash
# Qwen3.6-35B-A3B Quark W8A8 on Steve's pinned S2B vLLM/XPU snapshot.
# This is a research control, not a shelf entry. It intentionally mounts no
# Ornith compatibility code. P2P defaults off under the repository safety rule.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94}"
export CKPT="${CKPT:-/models/qwen3.6-35b-a3b/quark-w8a8-int8}"
export SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8-s2b-control}"
export NAME="${NAME:-qwen36_s2b_control}"
export PORT="${PORT:-18080}"
export QUANT="${QUANT:-quark}"
export TP="${TP:-2}"
export PP="${PP:-1}"
export GRAPH="${GRAPH:-1}"
export CGMODE="${CGMODE:-PIECEWISE}"
export ATTN="${ATTN:-}"
export MOE_BACKEND="${MOE_BACKEND:-auto}"
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-24}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export CAPSIZES="${CAPSIZES:-1,2}"
export P2PACCESS="${P2PACCESS:-0}"
export IPCX="${IPCX:-pidfd}"
export PUSH_AR="${PUSH_AR:-0}"
export PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-1}"
export PUSH_AR_MAXB="${PUSH_AR_MAXB:-67108864}"
export PUSH_AR_PREINIT="${PUSH_AR_PREINIT:-1}"
export PUSH_AR_PREINIT_STRICT="${PUSH_AR_PREINIT_STRICT:-1}"
export PUSH_AR_GRAPH_INPLACE="${PUSH_AR_GRAPH_INPLACE:-0}"
export EXACT_STEVE_CC="${EXACT_STEVE_CC:-0}"
export FORENSIC_VLLM_SRC="${FORENSIC_VLLM_SRC:-}"
export FORENSIC_SITECUSTOMIZE_HOST="${FORENSIC_SITECUSTOMIZE_HOST:-}"
export FORENSIC_FUSED_MOE_INTERFACE_HOST="${FORENSIC_FUSED_MOE_INTERFACE_HOST:-}"
export XPU_KERNEL_RUNTIME_HOST="${XPU_KERNEL_RUNTIME_HOST:-}"
export XPU_PROFILE="${XPU_PROFILE:-0}"
export XPU_PROFILE_DIR_HOST="${XPU_PROFILE_DIR_HOST:-}"
export NUMA_BIND="${NUMA_BIND:-0}"
export NUMA_BIND_CPUS="${NUMA_BIND_CPUS:-0-7,16-23|8-15,24-31}"
export IN="${IN:-512}"
export OUT="${OUT:-512}"
export CONC="${CONC:-1}"

BASE_SPLITOPS='"vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::linear_attention","vllm::plamo2_mamba_mixer","vllm::qwen_gdn_attention_core","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::kda_attention","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention"'
if [ "$EXACT_STEVE_CC" = 1 ]; then
  # Steve's accepted command supplied only cudagraph_mode=PIECEWISE. Let vLLM
  # choose its own default attention boundaries; do not inject our newer
  # repository split policy into this source-reproduction arm. CGMODE remains
  # parameterized for explicit boundary experiments; PIECEWISE is the control.
  export SPLITOPS=""
  export IGP=false
elif [ "$P2PACCESS" = 1 ]; then
  # Steve's path keeps communication in the forced graph. This mode remains
  # subject to the shared P2P wedge guard and is not the default.
  export SPLITOPS="${SPLITOPS:-$BASE_SPLITOPS}"
  export SYCLKERNELS="${SYCLKERNELS:-1}"
else
  # Host-staged oneCCL cannot be recorded by this stack. Capturable push-AR
  # owns all-reduce inside replay; all-gather remains an eager boundary for
  # this first one-factor transaction.
  if [ "$PUSH_AR" = 1 ] && [ "$PUSH_AR_GRAPH" = 1 ]; then
    P2P0_SPLITOPS="$BASE_SPLITOPS,\"vllm::all_gather\""
  else
    P2P0_SPLITOPS="$BASE_SPLITOPS,\"vllm::all_reduce\",\"vllm::all_gather\""
  fi
  export SPLITOPS="${SPLITOPS:-$P2P0_SPLITOPS}"
  export SYCLKERNELS="${SYCLKERNELS:-0}"
fi

export EXTRA_ARGS="${EXTRA_ARGS:---language-model-only --generation-config vllm --max-num-batched-tokens 8192 --uvicorn-log-level warning}"
case "$CGMODE" in
  PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
  *) echo "CGMODE must be PIECEWISE, FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE" >&2; exit 1 ;;
esac
case "$ATTN" in
  ''|TRITON_ATTN|FLASH_ATTN) ;;
  *) echo "ATTN must be empty, TRITON_ATTN, or FLASH_ATTN" >&2; exit 1 ;;
esac
[ -z "$ATTN" ] || export EXTRA_ARGS="$EXTRA_ARGS --attention-backend $ATTN"
case "$MOE_BACKEND" in
  auto|xpu|triton) ;;
  *) echo "MOE_BACKEND must be auto, xpu, or triton" >&2; exit 1 ;;
esac
[ "$MOE_BACKEND" = auto ] || export EXTRA_ARGS="$EXTRA_ARGS --moe-backend $MOE_BACKEND"
export MTPTOK=""
export SPEC=""

MOUNTS=(
  -v "$SCRIPT_DIR/qwen36_s2b_probe.py:/opt/b70_qwen36_s2b_probe.py:ro"
  -v "$SCRIPT_DIR/qwen36_steve_metric.py:/opt/b70_qwen36_steve_metric.py:ro"
  -v "$SCRIPT_DIR/qwen36_repeat_canary.py:/opt/b70_qwen36_repeat_canary.py:ro"
  -v "/mnt/vm_8tb/b70/results:/results"
)

if [ "$XPU_PROFILE" = 1 ]; then
  [ -n "$XPU_PROFILE_DIR_HOST" ] || {
    echo "XPU_PROFILE=1 requires XPU_PROFILE_DIR_HOST" >&2
    exit 1
  }
  mkdir -p "$XPU_PROFILE_DIR_HOST"
  MOUNTS+=( -v "$XPU_PROFILE_DIR_HOST:/profiles" )
  export EXTRA_ARGS="$EXTRA_ARGS --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/profiles\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true,\"delay_iterations\":2,\"max_iterations\":8}"
elif [ "$XPU_PROFILE" != 0 ]; then
  echo "XPU_PROFILE must be 0 or 1" >&2
  exit 1
fi

NUMA_DOCKER_ARGS=()
if [ "$NUMA_BIND" = 1 ]; then
  IFS='|' read -r numa_rank0_cpus numa_rank1_cpus numa_extra <<< "$NUMA_BIND_CPUS"
  [ -n "$numa_rank0_cpus" ] && [ -n "$numa_rank1_cpus" ] && [ -z "$numa_extra" ] || {
    echo "NUMA_BIND_CPUS must contain exactly two CPU lists separated by |" >&2
    exit 1
  }
  export EXTRA_ARGS="$EXTRA_ARGS --numa-bind --numa-bind-nodes 0 0 --numa-bind-cpus $numa_rank0_cpus $numa_rank1_cpus"
  NUMA_DOCKER_ARGS+=( --cap-add SYS_NICE )
elif [ "$NUMA_BIND" != 0 ]; then
  echo "NUMA_BIND must be 0 or 1" >&2
  exit 1
fi
DOCKER_ENV=(
  -e VLLM_USE_V1=1
  -e VLLM_TARGET_DEVICE=xpu
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e XPU_GRAPH=1
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1
  -e VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
  -e VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
  -e VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1
  -e VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
  -e VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1
  -e VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1
  -e VLLM_XPU_QUARK_W8A8_MOE=1
  -e VLLM_XPU_FORCE_QUARK_REPACK=0
  -e VLLM_XPU_INT8_MOE_MIXED_WORKSPACE=1
  -e VLLM_XPU_GDN_REUSE_QKVZ_BA_QUANT=clone
  -e VLLM_XPU_GDN_NATIVE_FALLBACK=prefill
  -e VLLM_XPU_GDN_PREFILL_RECURRENT_FALLBACK=1
  -e VLLM_XPU_ZERO_FRESH_GDN_STATE=1
  -e VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY=1
  -e VLLM_XPU_GREEDY_SAMPLE_TOPK_FALLBACK=1
  # Steve pinned eth1 on bare metal. The isolated Docker equivalent is eth0.
  -e FI_TCP_IFACE=eth0
  -e CCL_KVS_IFACE=eth0
)

if [ -n "$FORENSIC_VLLM_SRC" ]; then
  [ "$EXACT_STEVE_CC" = 1 ] || {
    echo "FORENSIC_VLLM_SRC requires EXACT_STEVE_CC=1" >&2
    exit 1
  }
  [ -f "$FORENSIC_VLLM_SRC/vllm/__init__.py" ] || {
    echo "Invalid FORENSIC_VLLM_SRC: $FORENSIC_VLLM_SRC" >&2
    exit 1
  }
  MOUNTS+=( -v "$FORENSIC_VLLM_SRC:/opt/forensic_vllm:ro" )
  if [ -n "$FORENSIC_SITECUSTOMIZE_HOST" ]; then
    [ -f "$FORENSIC_SITECUSTOMIZE_HOST" ] || {
      echo "Invalid FORENSIC_SITECUSTOMIZE_HOST: $FORENSIC_SITECUSTOMIZE_HOST" >&2
      exit 1
    }
    MOUNTS+=(
      -v "$FORENSIC_SITECUSTOMIZE_HOST:/opt/forensic_site/sitecustomize.py:ro"
    )
    DOCKER_ENV+=( -e PYTHONPATH=/opt/forensic_site:/opt/forensic_vllm )
  else
    DOCKER_ENV+=( -e PYTHONPATH=/opt/forensic_vllm )
  fi
else
  MOUNTS+=(
    -v "$SCRIPT_DIR/qwen36_s2b_sitecustomize.py:/opt/b70_qwen36_site/sitecustomize.py:ro"
  )
  DOCKER_ENV+=( -e PYTHONPATH=/opt/b70_qwen36_site )
fi

if [ -n "$XPU_KERNEL_RUNTIME_HOST" ]; then
  for required in \
    vllm_xpu_kernels/__init__.py \
    vllm_xpu_kernels/_C.abi3.so \
    vllm_xpu_kernels/_moe_C.abi3.so \
    vllm_xpu_kernels/_xpu_C.abi3.so \
    vllm_xpu_kernels/libgrouped_gemm_xe_2.so \
    vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so \
    vllm_xpu_kernels/fused_moe_interface.py; do
    [ -f "$XPU_KERNEL_RUNTIME_HOST/$required" ] || {
      echo "Incomplete XPU kernel runtime: missing $required" >&2
      exit 1
    }
  done
  MOUNTS+=( -v "$XPU_KERNEL_RUNTIME_HOST:/opt/june-runtime:ro" )
  if [ -n "$FORENSIC_FUSED_MOE_INTERFACE_HOST" ]; then
    [ -f "$FORENSIC_FUSED_MOE_INTERFACE_HOST" ] || {
      echo "Invalid FORENSIC_FUSED_MOE_INTERFACE_HOST: $FORENSIC_FUSED_MOE_INTERFACE_HOST" >&2
      exit 1
    }
    MOUNTS+=(
      -v "$FORENSIC_FUSED_MOE_INTERFACE_HOST:/opt/june-runtime/vllm_xpu_kernels/fused_moe_interface.py:ro"
    )
  fi
  if [ -n "$FORENSIC_VLLM_SRC" ]; then
    if [ -n "$FORENSIC_SITECUSTOMIZE_HOST" ]; then
      DOCKER_ENV+=(
        -e PYTHONPATH=/opt/forensic_site:/opt/forensic_vllm:/opt/june-runtime
      )
    else
      DOCKER_ENV+=( -e PYTHONPATH=/opt/forensic_vllm:/opt/june-runtime )
    fi
  else
    DOCKER_ENV+=( -e PYTHONPATH=/opt/june-runtime:/opt/b70_qwen36_site )
  fi
fi

EXACT_POST_ENV=()
if [ "$EXACT_STEVE_CC" = 1 ]; then
  # lib.sh normally forces pidfd. Steve's accepted launcher deliberately
  # removed both settings and let oneCCL select its defaults. These trailing
  # name-only entries override the earlier Docker environment assignments.
  EXACT_POST_ENV+=( -e CCL_ZE_IPC_EXCHANGE -e CCL_WORKER_COUNT )
fi

if [ "$PUSH_AR" = 1 ]; then
  PUSH_AR_DIR="$SCRIPT_DIR/../contrib/vllm_push_allreduce"
  PUSH_AR_SO_HOST="${PUSH_AR_SO_HOST:-$PUSH_AR_DIR/prebuilt/libxpu_push_ar_graph.so}"
  [ "$TP" = 2 ] || { echo "PUSH_AR=1 requires TP=2" >&2; exit 1; }
  [ "$GRAPH" = 1 ] && [ "$PUSH_AR_GRAPH" = 1 ] || {
    echo "This control's PUSH_AR=1 arm requires GRAPH=1 PUSH_AR_GRAPH=1" >&2
    exit 1
  }
  [ -f "$PUSH_AR_SO_HOST" ] || {
    echo "Missing capturable push-AR SO: $PUSH_AR_SO_HOST" >&2
    exit 1
  }
  MOUNTS+=(
    -v "$PUSH_AR_DIR:/opt/push_ar:ro"
    -v "$PUSH_AR_SO_HOST:/opt/push_ar_runtime/libxpu_push_ar_graph.so:ro"
  )
  push_ar_pythonpath=/opt/push_ar
  push_ar_chain=
  if [ -n "$FORENSIC_VLLM_SRC" ]; then
    push_ar_pythonpath="$push_ar_pythonpath:/opt/forensic_vllm"
    [ -z "$FORENSIC_SITECUSTOMIZE_HOST" ] || \
      push_ar_chain=/opt/forensic_site/sitecustomize.py
  else
    push_ar_chain=/opt/b70_qwen36_site/sitecustomize.py
  fi
  [ -z "$XPU_KERNEL_RUNTIME_HOST" ] || \
    push_ar_pythonpath="$push_ar_pythonpath:/opt/june-runtime"
  DOCKER_ENV+=(
    -e PYTHONPATH="$push_ar_pythonpath"
    -e PUSH_AR_CHAIN_SITECUSTOMIZE="$push_ar_chain"
    -e PUSH_AR_SO=/opt/push_ar_runtime/libxpu_push_ar_graph.so
    -e PUSH_AR_DISABLE=0
    -e PUSH_AR_GRAPH=1
    -e PUSH_AR_MIN_NUMEL=0
    -e PUSH_AR_MAXB="$PUSH_AR_MAXB"
    -e PUSH_AR_PREINIT="$PUSH_AR_PREINIT"
    -e PUSH_AR_PREINIT_STRICT="$PUSH_AR_PREINIT_STRICT"
    -e PUSH_AR_GRAPH_INPLACE="$PUSH_AR_GRAPH_INPLACE"
  )
fi

if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for setting in ${B70_EXTRA_ENV}; do
    DOCKER_ENV+=( -e "$setting" )
  done
fi

source "$SCRIPT_DIR/../../rdy_to_serve/_common/lib.sh"
CACHE_DIR_HOST="${CACHE_DIR_HOST:-$ROOT/vllm_cache}"
mkdir -p "$CACHE_DIR_HOST"

b70_serve() {
  b70_build
  if [ "$EXACT_STEVE_CC" = 1 ]; then
    local exact_cc_replaced=0
    for ((arg_i=0; arg_i<${#ARGS[@]}; arg_i++)); do
      if [ "${ARGS[$arg_i]}" = "--compilation-config" ]; then
        ARGS[$((arg_i + 1))]="{\"cudagraph_mode\":\"$CGMODE\"}"
        exact_cc_replaced=1
        break
      fi
    done
    [ "$exact_cc_replaced" = 1 ] || {
      echo "EXACT_STEVE_CC could not find --compilation-config" >&2
      return 1
    }
  fi
  docker rm -f "$NAME" 2>/dev/null || true
  echo "=== serve $SERVED  IMG=$IMG  TP=$TP PP=$PP  GRAPH=$GRAPH  port=$PORT ==="
  echo "vllm ${ARGS[*]}"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" "${GDOCK[@]}" \
    --cap-add SYS_PTRACE "${NUMA_DOCKER_ARGS[@]+"${NUMA_DOCKER_ARGS[@]}"}" \
    --security-opt seccomp=unconfined \
    -v "$MODELS_FILES:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$CACHE_DIR_HOST:/vllm_cache" \
    -v "$ROOT/tmp_ssd:/tmp_ssd" "${MOUNTS[@]+"${MOUNTS[@]}"}" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    "${DOCKER_ENV[@]+"${DOCKER_ENV[@]}"}" \
    "${MGPU[@]}" "${GENV[@]}" "${EXACT_POST_ENV[@]+"${EXACT_POST_ENV[@]}"}" \
    --entrypoint vllm "$IMG" "${ARGS[@]}" >/dev/null
}

b70_gen_probe() {
  echo "--- Qwen S2B semantic/repetition probe ---"
  docker exec "$NAME" python /opt/b70_qwen36_s2b_probe.py \
    --base-url "http://127.0.0.1:$PORT" --model "$SERVED"
}

b70_bench() {
  local stamp out out_dir json_out color_out profile_out profile_rc
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  out_dir="/results/logs/$NAME"
  out="$out_dir/steve_metric_${stamp}.json"
  json_out="$out_dir/json_repeat16_${stamp}.json"
  color_out="$out_dir/color_repeat16_${stamp}.json"
  if [ "$XPU_PROFILE" = 1 ]; then
    profile_out="$out_dir/xpu_profile_workload_${stamp}.json"
    echo "--- bounded XPU profile: skip prefill, record eight decode iterations ---"
    curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" || return $?
    docker exec "$NAME" python /opt/b70_qwen36_steve_metric.py \
      --base-url "http://127.0.0.1:$PORT" \
      --model "$SERVED" \
      --tokenizer "$CKPT" \
      --out "$profile_out"
    profile_rc=$?
    curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" || profile_rc=$?
    [ "$profile_rc" = 0 ] || return "$profile_rc"
    compgen -G "$XPU_PROFILE_DIR_HOST/*.json.gz" >/dev/null || {
      echo "XPU profiler produced no trace" >&2
      return 1
    }
    echo "host_profile_artifact=$ROOT/results/logs/$NAME/$(basename "$profile_out")"
  fi
  echo "--- exact Steve-shaped natural-chat p512/o512 metric ---"
  docker exec "$NAME" python /opt/b70_qwen36_steve_metric.py \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED" \
    --tokenizer "$CKPT" \
    --out "$out" || return $?
  echo "--- Steve fixed-ChatML JSON repeat canary 16/16 ---"
  docker exec "$NAME" python /opt/b70_qwen36_repeat_canary.py \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED" \
    --tokenizer "$CKPT" \
    --case json \
    --repeats 16 \
    --stop-on-mismatch \
    --output-json "$json_out" || return $?
  echo "--- Steve fixed-ChatML color repeat canary 16/16 ---"
  docker exec "$NAME" python /opt/b70_qwen36_repeat_canary.py \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED" \
    --tokenizer "$CKPT" \
    --case color \
    --repeats 16 \
    --stop-on-mismatch \
    --output-json "$color_out" || return $?
  echo "host_json_canary=/mnt/vm_8tb/b70/results${json_out#/results}"
  echo "host_color_canary=/mnt/vm_8tb/b70/results${color_out#/results}"
  echo "host_artifact=/mnt/vm_8tb/b70/results${out#/results}"
}

b70_dispatch "$@"
