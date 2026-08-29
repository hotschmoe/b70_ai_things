#!/usr/bin/env bash
# Refreshed-stack research serve for Ornith-1.5-35B-A3B Quark-compatible
# W8A8 RTN plus the trained Shisa MTP head. Wrap GPU actions with bin/gpu-run.
set -uo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78}"
NAME="${NAME:-sglang_ornith15_w8a8_refresh}"
CKPT="${CKPT:-/models/ornith-1.5-35b-a3b/w8a8-rtn-mtp-shisa}"
SERVED="${SERVED:-ornith-1.5-35b-a3b-W8A8-rtn-shisa}"
PORT="${PORT:-18080}"
TP="${TP:-2}"
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
MAXREQ="${MAXREQ:-4}"
GRAPH_BS="${GRAPH_BS:-1 2 4}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
# Bound the Level Zero command-list growth caused by repeated XPUGraph replay.
# The matched 12-batch soak selected 500; set 0 only for the retained control.
CG_RECLAIM="${CG_RECLAIM:-500}"
DENSE_NATIVE="${DENSE_NATIVE:-0}"
MTP="${MTP:-0}"
SPEC_STEPS="${SPEC_STEPS:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-2}"
DECODE_GRAPH="${DECODE_GRAPH:-breakable}"
TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
THINKCAP="${THINKCAP-}"
if [ -z "${SYCL_KERNELS+x}" ]; then
  if [ "$DECODE_GRAPH" = 1 ] || [ "$DECODE_GRAPH" = full ]; then
    SYCL_KERNELS=1
  else
    SYCL_KERNELS=0
  fi
fi
IPC_EXCHANGE="${IPC_EXCHANGE:-pidfd}"
LOG="${LOG:-$ROOT/sglang_ornith15_w8a8_refresh.log}"
SP=/opt/venv/lib/python3.12/site-packages
QUARK_PATCH="$REPO/sglang/patches/quark_moe_int8.py"
ACTQ_PATCH="$REPO/sglang/patches/int8_actquant_xpu.py"
W8A8_PATCH="$REPO/sglang/refresh/b70_xpu_w8a8.py"
PTH="$REPO/sglang/refresh/b70_ornith_w8a8.pth"

say() { echo "[$(date +%H:%M:%S)] $*"; }

preflight() {
  docker image inspect "$IMG" >/dev/null 2>&1 || {
    say "missing image: $IMG"
    return 1
  }
  [ -f "$REPO/models/files/${CKPT#/models/}/model.safetensors.index.json" ] || {
    say "missing checkpoint: $REPO/models/files/${CKPT#/models/}"
    return 1
  }
  for path in "$QUARK_PATCH" "$ACTQ_PATCH" "$W8A8_PATCH" "$PTH"; do
    [ -f "$path" ] || { say "missing patch: $path"; return 1; }
  done
  [ "$TP" = 2 ] || {
    say "this checkpoint is qualified only with TP=2, got TP=$TP"
    return 1
  }
  case "$MTP" in
    0|1) ;;
    *) say "MTP must be 0 or 1, got $MTP"; return 1 ;;
  esac
  case "$DENSE_NATIVE" in
    0|1) ;;
    *) say "DENSE_NATIVE must be 0 or 1, got $DENSE_NATIVE"; return 1 ;;
  esac
  if [ "$MTP" = 1 ]; then
    [ "$SPEC_STEPS" -ge 1 ] 2>/dev/null || {
      say "SPEC_STEPS must be a positive integer, got $SPEC_STEPS"
      return 1
    }
    [ "$SPEC_DRAFT" -eq $((SPEC_STEPS + 1)) ] || {
      say "topk=1 NEXTN requires SPEC_DRAFT=SPEC_STEPS+1"
      return 1
    }
    case "$SERVED" in
      *nextn*) ;;
      *) say "MTP=1 requires a served ID containing nextn, got $SERVED"; return 1 ;;
    esac
    [ -f "$REPO/models/files/${CKPT#/models/}/model-mtp.safetensors" ] || {
      say "MTP=1 but model-mtp.safetensors is missing"
      return 1
    }
  fi
  case "$DECODE_GRAPH" in
    0|1|full|breakable) ;;
    *) say "DECODE_GRAPH must be 0, 1, full, or breakable"; return 1 ;;
  esac
  local graph_bs
  for graph_bs in $GRAPH_BS; do
    [ "$graph_bs" -ge 1 ] 2>/dev/null || {
      say "GRAPH_BS must contain positive integers, got $GRAPH_BS"
      return 1
    }
  done
  [ "$CHUNKED_PREFILL" -ge 1 ] 2>/dev/null || {
    say "CHUNKED_PREFILL must be a positive integer, got $CHUNKED_PREFILL"
    return 1
  }
  [ "$CG_RECLAIM" -ge 0 ] 2>/dev/null || {
    say "CG_RECLAIM must be a nonnegative integer, got $CG_RECLAIM"
    return 1
  }
  case "$SYCL_KERNELS" in
    0|1) ;;
    *) say "SYCL_KERNELS must be 0 or 1"; return 1 ;;
  esac
  case "$IPC_EXCHANGE" in
    pidfd|drmfd|sockets) ;;
    *) say "IPC_EXCHANGE must be pidfd, drmfd, or sockets"; return 1 ;;
  esac
  case "$TOOLPARSER" in
    qwen3_coder|none) ;;
    *) say "TOOLPARSER must be qwen3_coder or none, got $TOOLPARSER"; return 1 ;;
  esac
  if [ -n "$THINKCAP" ] && ! { [ "$THINKCAP" -ge 1 ] 2>/dev/null; }; then
    say "THINKCAP must be empty or a positive integer, got $THINKCAP"
    return 1
  fi
}

start() {
  preflight || return 1
  mkdir -p "$ROOT/hf_cache" "$ROOT/sgl_cache/inductor" "$ROOT/sgl_cache/triton"
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  local graph_args
  local security_args=()
  local grammar_args="--grammar-backend none"
  local spec_args=""
  local tool_args=""
  local think_env=()
  local breakable_graph=0
  if [ "$DECODE_GRAPH" = 1 ] || [ "$DECODE_GRAPH" = full ]; then
    graph_args="--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode $GRAPH_BS"
    # oneCCL needs pidfd_getfd for graph-owned allocations. Keep these
    # permissions scoped to the FULL graph research route.
    security_args=(--cap-add SYS_PTRACE --security-opt seccomp=unconfined)
  elif [ "$DECODE_GRAPH" = breakable ]; then
    graph_args="--cuda-graph-backend-decode breakable --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode $GRAPH_BS"
    breakable_graph=1
  else
    graph_args="--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled"
  fi
  if [ "$MTP" = 1 ]; then
    spec_args="--speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --speculative-draft-model-quantization unquant"
  fi
  if [ "$TOOLPARSER" != none ]; then
    tool_args="--tool-call-parser $TOOLPARSER"
  fi
  if [ -n "$THINKCAP" ]; then
    think_env=(-e "SGLANG_MAX_THINK_TOKENS=$THINKCAP")
    grammar_args="--grammar-backend xgrammar --enable-strict-thinking"
  fi

  say "serve image=$IMG model=$SERVED tp=$TP ctx=$CTX dense_native=$DENSE_NATIVE mtp=$MTP spec_steps=$SPEC_STEPS graph=$DECODE_GRAPH graph_bs=$GRAPH_BS graph_reclaim=$CG_RECLAIM chunked_prefill=$CHUNKED_PREFILL maxreq=$MAXREQ tool_parser=$TOOLPARSER think_cap=${THINKCAP:-unlimited} sycl_kernels=$SYCL_KERNELS"
  docker run -d --name "$NAME" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    "${security_args[@]}" \
    -p "$PORT:$PORT" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" \
    -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$QUARK_PATCH:$SP/quark_moe_int8.py:ro" \
    -v "$ACTQ_PATCH:$SP/int8_actquant_xpu.py:ro" \
    -v "$W8A8_PATCH:$SP/b70_xpu_w8a8.py:ro" \
    -v "$PTH:$SP/b70_ornith_w8a8.pth:ro" \
    -e HF_HOME=/hf_cache \
    -e XDG_CACHE_HOME=/sgl_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e TRITON_CACHE_DIR=/sgl_cache/triton \
    -e B70_XPU_W8A8=1 \
    -e B70_XPU_W8A8_NATIVE="$DENSE_NATIVE" \
    -e B70_QUARK_MOE_INT8_AUTOINSTALL=1 \
    -e B70_QUARK_DENSE_NATIVE="$DENSE_NATIVE" \
    -e B70_XPU_MTP="$MTP" \
    -e B70_XPU_BREAKABLE_GRAPH="$breakable_graph" \
    -e B70_XPU_CG_RECLAIM="$CG_RECLAIM" \
    "${think_env[@]}" \
    -e VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY="$DENSE_NATIVE" \
    -e VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER="$DENSE_NATIVE" \
    -e CCL_ATL_TRANSPORT=ofi \
    -e CCL_ENABLE_SYCL_KERNELS="$SYCL_KERNELS" \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e CCL_TOPO_P2P_ACCESS=0 \
    -e CCL_ZE_IPC_EXCHANGE="$IPC_EXCHANGE" \
    -e FI_TCP_IFACE=eth0 \
    -e CCL_KVS_IFACE=eth0 \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    "$IMG" bash -c "exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' \
      --trust-remote-code --device xpu --dtype bfloat16 \
      --kv-cache-dtype bfloat16 \
      --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 $grammar_args \
      $graph_args $spec_args $tool_args \
      --disable-radix-cache --disable-overlap-schedule --skip-server-warmup \
      --chunked-prefill-size '$CHUNKED_PREFILL' \
      --disable-custom-all-reduce --reasoning-parser qwen3 --tp-size '$TP' \
      --context-length '$CTX' --mem-fraction-static '$MEMFRAC' \
      --max-running-requests '$MAXREQ' --host 0.0.0.0 --port '$PORT'" \
    >/dev/null

  local healthy=0
  for i in $(seq 1 240); do
    if ! docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | rg -qx "$NAME"; then
      say "server exited"
      docker logs "$NAME" >"$LOG" 2>&1 || true
      tail -n 120 "$LOG"
      return 1
    fi
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || true)" = 200 ]; then
      healthy=1
      say "health 200 after about $((i * 5)) seconds"
      break
    fi
    sleep 5
  done
  docker logs "$NAME" >"$LOG" 2>&1 || true
  [ "$healthy" = 1 ] || { say "health timeout; see $LOG"; return 1; }

  curl -fsS "http://localhost:$PORT/v1/models" | python3 -c \
    "import json,sys; ids=[x['id'] for x in json.load(sys.stdin)['data']]; assert '$SERVED' in ids, ids; print('identity ->', ids)" || return 1
  say "healthy with exact model identity; endpoint left running"
}

stop() {
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  say "stopped $NAME"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  logs) docker logs -f "$NAME" ;;
  status)
    docker ps --filter "name=^/${NAME}$" --format '{{.Names}} {{.Status}}'
    curl -fsS "http://localhost:$PORT/health" && echo
    ;;
  *) echo "usage: $0 {start|stop|logs|status}"; exit 2 ;;
esac
