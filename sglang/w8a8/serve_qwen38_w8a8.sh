#!/usr/bin/env bash
# Conservative refreshed-stack bring-up for Qwen3.8-27B compressed-tensors
# W8A8 GPTQ on two Intel Arc Pro B70 cards. Wrap GPU actions with bin/gpu-run.
set -uo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd}"
NAME="${NAME:-sglang_qwen38_w8a8_refresh}"
CKPT="${CKPT:-/models/qwen3.8-27b/w8a8-gptq}"
SERVED="${SERVED:-qwen3.8-27b-W8A8-gptq}"
PORT="${PORT:-18080}"
TP="${TP:-2}"
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
MAXREQ="${MAXREQ:-4}"
NATIVE="${NATIVE:-0}"
ONEDNN_INPUT_DEP="${ONEDNN_INPUT_DEP:-$NATIVE}"
ONEDNN_BARRIER="${ONEDNN_BARRIER:-$NATIVE}"
DECODE_GRAPH="${DECODE_GRAPH:-0}"
if [ -z "${SYCL_KERNELS+x}" ]; then
  if [ "$DECODE_GRAPH" = 1 ] || [ "$DECODE_GRAPH" = full ]; then
    SYCL_KERNELS=1
  else
    SYCL_KERNELS=0
  fi
fi
IPC_EXCHANGE="${IPC_EXCHANGE:-pidfd}"
LOG="${LOG:-$ROOT/sglang_qwen38_w8a8_refresh.log}"

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
  [ "$TP" = 2 ] || {
    say "this checkpoint is qualified only with TP=2, got TP=$TP"
    return 1
  }
  case "$NATIVE" in
    0|1) ;;
    *) say "NATIVE must be 0 or 1, got $NATIVE"; return 1 ;;
  esac
  case "$ONEDNN_BARRIER" in
    0|1) ;;
    *) say "ONEDNN_BARRIER must be 0 or 1, got $ONEDNN_BARRIER"; return 1 ;;
  esac
  case "$ONEDNN_INPUT_DEP" in
    0|1) ;;
    *) say "ONEDNN_INPUT_DEP must be 0 or 1, got $ONEDNN_INPUT_DEP"; return 1 ;;
  esac
  case "$DECODE_GRAPH" in
    0|1|full|breakable) ;;
    *) say "DECODE_GRAPH must be 0, 1, full, or breakable, got $DECODE_GRAPH"; return 1 ;;
  esac
  case "$SYCL_KERNELS" in
    0|1) ;;
    *) say "SYCL_KERNELS must be 0 or 1, got $SYCL_KERNELS"; return 1 ;;
  esac
  case "$IPC_EXCHANGE" in
    pidfd|drmfd|sockets) ;;
    *) say "IPC_EXCHANGE must be pidfd, drmfd, or sockets, got $IPC_EXCHANGE"; return 1 ;;
  esac
}

start() {
  preflight || return 1
  mkdir -p "$ROOT/hf_cache" "$ROOT/sgl_cache/inductor" "$ROOT/sgl_cache/triton"
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  local graph_args
  local breakable_graph=0
  if [ "$DECODE_GRAPH" = 1 ] || [ "$DECODE_GRAPH" = full ]; then
    graph_args="--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode 1 2 4"
  elif [ "$DECODE_GRAPH" = breakable ]; then
    graph_args="--cuda-graph-backend-decode breakable --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode 1 2 4"
    breakable_graph=1
  else
    graph_args="--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled"
  fi

  say "serve image=$IMG model=$SERVED tp=$TP ctx=$CTX memfrac=$MEMFRAC native=$NATIVE onednn_input_dep=$ONEDNN_INPUT_DEP onednn_barrier=$ONEDNN_BARRIER decode_graph=$DECODE_GRAPH sycl_kernels=$SYCL_KERNELS ipc_exchange=$IPC_EXCHANGE"
  docker run -d --name "$NAME" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    -p "$PORT:$PORT" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" \
    -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache \
    -e XDG_CACHE_HOME=/sgl_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e TRITON_CACHE_DIR=/sgl_cache/triton \
    -e B70_XPU_W8A8=1 \
    -e B70_XPU_W8A8_NATIVE="$NATIVE" \
    -e B70_XPU_BREAKABLE_GRAPH="$breakable_graph" \
    -e VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY="$ONEDNN_INPUT_DEP" \
    -e VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER="$ONEDNN_BARRIER" \
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
      --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --grammar-backend none \
      $graph_args \
      --disable-radix-cache --disable-overlap-schedule --skip-server-warmup \
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
  [ "$healthy" = 1 ] || {
    say "health timeout; see $LOG"
    return 1
  }

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
