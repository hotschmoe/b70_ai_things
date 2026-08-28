#!/usr/bin/env bash
# Current SGLang TP=2 research serve for Qwen3.8-27B RadixArk NVFP4.
# Wrap all GPU actions with bin/gpu-run.
set -uo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78}"
NAME="${NAME:-sglang_qwen38_nvfp4_refresh}"
CKPT="${CKPT:-/models/qwen3.8-27b/nvfp4-radixark}"
SERVED="${SERVED:-qwen3.8-27b-NVFP4-radixark-sglang-current}"
PORT="${PORT:-18182}"
TP="${TP:-2}"
CTX="${CTX:-4096}"
MEMFRAC="${MEMFRAC:-0.80}"
MAXREQ="${MAXREQ:-1}"
GRAPH_BS="${GRAPH_BS:-1}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-128}"
DECODE_GRAPH="${DECODE_GRAPH:-full}"
TOOLPARSER="${TOOLPARSER:-none}"
THINKCAP="${THINKCAP-}"
F8_SCALE_M_MAX="${F8_SCALE_M_MAX:-8}"
FP8_W8A16_M_MAX="${FP8_W8A16_M_MAX:-1}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
LINEAR_ATTN_PREFILL_BACKEND="${LINEAR_ATTN_PREFILL_BACKEND:-triton}"
IPC_EXCHANGE="${IPC_EXCHANGE:-pidfd}"
HOST_MEM_MIN_GIB="${HOST_MEM_MIN_GIB:-64}"
if [ -z "${SYCL_KERNELS+x}" ]; then
  if [ "$DECODE_GRAPH" = full ]; then
    SYCL_KERNELS=1
  else
    SYCL_KERNELS=0
  fi
fi
LOG="${LOG:-$ROOT/sglang_qwen38_nvfp4_refresh.log}"

SP=/opt/venv/lib/python3.12/site-packages
NVFP4_SO="${NVFP4_SO:-$ROOT/nvfp4_kernel_v028/_xpu_C.abi3.so}"
GDN_SO="${GDN_SO:-$ROOT/nvfp4_kernel_v028/libgdn_attn_kernels_xe_2.so}"
PATCH="$REPO/sglang/refresh/b70_xpu_nvfp4.py"
PTH="$REPO/sglang/refresh/b70_xpu_nvfp4.pth"
RUNTIME_CONFIG="$ROOT/sglang_nvfp4_runtime/qwen38-config.json"
RUNTIME_HF_QUANT="$ROOT/sglang_nvfp4_runtime/qwen38-hf-quant-config.json"

say() { echo "[$(date +%H:%M:%S)] $*"; }

preflight() {
  docker image inspect "$IMG" >/dev/null 2>&1 || {
    say "missing image: $IMG"
    return 1
  }
  local host_ckpt="$REPO/models/files/${CKPT#/models/}"
  [ -f "$host_ckpt/model.safetensors.index.json" ] || {
    say "missing checkpoint: $host_ckpt"
    return 1
  }
  for path in "$NVFP4_SO" "$GDN_SO" "$PATCH" "$PTH"; do
    [ -f "$path" ] || { say "missing dependency: $path"; return 1; }
  done
  [ "$TP" = 2 ] || { say "this route requires TP=2"; return 1; }
  [ "$MAXREQ" = 1 ] || { say "single-stream qualification requires MAXREQ=1"; return 1; }
  case "$DECODE_GRAPH" in
    0|full) ;;
    *) say "DECODE_GRAPH must be 0 or full"; return 1 ;;
  esac
  case "$IPC_EXCHANGE" in
    pidfd|sockets|drmfd) ;;
    *) say "IPC_EXCHANGE must be pidfd, sockets, or drmfd"; return 1 ;;
  esac
  case "$SYCL_KERNELS" in
    0|1) ;;
    *) say "SYCL_KERNELS must be 0 or 1"; return 1 ;;
  esac
  case "$LINEAR_ATTN_BACKEND" in
    triton|intel_xpu) ;;
    *) say "LINEAR_ATTN_BACKEND must be triton or intel_xpu"; return 1 ;;
  esac
  case "$LINEAR_ATTN_PREFILL_BACKEND" in
    triton|intel_xpu) ;;
    *) say "LINEAR_ATTN_PREFILL_BACKEND must be triton or intel_xpu"; return 1 ;;
  esac
  case "$TOOLPARSER" in
    qwen3_coder|none) ;;
    *) say "TOOLPARSER must be qwen3_coder or none, got $TOOLPARSER"; return 1 ;;
  esac
  if [ -n "$THINKCAP" ] && ! { [ "$THINKCAP" -ge 1 ] 2>/dev/null; }; then
    say "THINKCAP must be empty or a positive integer, got $THINKCAP"
    return 1
  fi
  [ "$CHUNKED_PREFILL" -ge 1 ] 2>/dev/null || {
    say "CHUNKED_PREFILL must be positive"
    return 1
  }
  [ "$F8_SCALE_M_MAX" -ge 0 ] 2>/dev/null || {
    say "F8_SCALE_M_MAX must be nonnegative"
    return 1
  }
  [ "$FP8_W8A16_M_MAX" -ge 0 ] 2>/dev/null || {
    say "FP8_W8A16_M_MAX must be nonnegative"
    return 1
  }

  local available_kib
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [ "$available_kib" -lt $((HOST_MEM_MIN_GIB * 1024 * 1024)) ]; then
    say "host memory gate failed: MemAvailable=$((available_kib / 1024 / 1024)) GiB"
    return 1
  fi

  mkdir -p "$(dirname "$RUNTIME_CONFIG")"
  python3 - \
    "$host_ckpt/config.json" "$RUNTIME_CONFIG" \
    "$host_ckpt/hf_quant_config.json" "$RUNTIME_HF_QUANT" <<'PY'
import json
import sys

config_source, config_destination, hf_source, hf_destination = sys.argv[1:]
with open(config_source, encoding="utf-8") as handle:
    config = json.load(handle)
config.get("quantization_config", {}).pop("kv_cache_scheme", None)
config["language_model_only"] = True
with open(config_destination, "w", encoding="ascii") as handle:
    json.dump(config, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")

with open(hf_source, encoding="utf-8") as handle:
    hf_quant = json.load(handle)
quantization = hf_quant.get("quantization", hf_quant)
quantization.pop("kv_cache_quant_algo", None)
quantization.pop("kv_cache_scheme", None)
with open(hf_destination, "w", encoding="ascii") as handle:
    json.dump(hf_quant, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
}

start() {
  preflight || return 1
  local cache_root="$ROOT/sgl_cache/qwen38_nvfp4_adc915d_96e33b4e"
  mkdir -p "$ROOT/hf_cache" "$cache_root/inductor" "$cache_root/triton"
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  local graph_args
  local security_args=()
  local tool_args=""
  local grammar_args="--grammar-backend none"
  local think_env=()
  if [ "$DECODE_GRAPH" = full ]; then
    graph_args="--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode $GRAPH_BS"
    security_args=(--cap-add SYS_PTRACE --security-opt seccomp=unconfined)
  else
    graph_args="--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled"
  fi
  if [ "$TOOLPARSER" != none ]; then
    tool_args="--tool-call-parser $TOOLPARSER"
  fi
  if [ -n "$THINKCAP" ]; then
    think_env=(-e "SGLANG_MAX_THINK_TOKENS=$THINKCAP")
    grammar_args="--grammar-backend xgrammar --enable-strict-thinking"
  fi

  say "serve image=$IMG model=$SERVED tp=$TP graph=$DECODE_GRAPH graph_bs=$GRAPH_BS chunk=$CHUNKED_PREFILL maxreq=$MAXREQ f8_scale_m_max=$F8_SCALE_M_MAX fp8_w8a16_m_max=$FP8_W8A16_M_MAX linear_attn=$LINEAR_ATTN_BACKEND linear_attn_prefill=$LINEAR_ATTN_PREFILL_BACKEND tool_parser=$TOOLPARSER think_cap=${THINKCAP:-unlimited} p2p=0 ipc=$IPC_EXCHANGE sycl_kernels=$SYCL_KERNELS"
  docker run -d --name "$NAME" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    "${security_args[@]}" \
    -p "$PORT:$PORT" \
    -v "$REPO/models/files:/models:ro" \
    -v "$RUNTIME_CONFIG:$CKPT/config.json:ro" \
    -v "$RUNTIME_HF_QUANT:$CKPT/hf_quant_config.json:ro" \
    -v "$ROOT/hf_cache:/hf_cache" \
    -v "$cache_root:/sgl_cache" \
    -v "$NVFP4_SO:$SP/vllm_xpu_kernels/_xpu_C.abi3.so:ro" \
    -v "$GDN_SO:$SP/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so:ro" \
    -v "$PATCH:$SP/b70_xpu_nvfp4.py:ro" \
    -v "$PTH:$SP/b70_xpu_nvfp4.pth:ro" \
    -e HF_HOME=/hf_cache \
    -e XDG_CACHE_HOME=/sgl_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e TRITON_CACHE_DIR=/sgl_cache/triton \
    -e B70_XPU_NVFP4=1 \
    -e B70_NVFP4_F8_SCALE_M_MAX="$F8_SCALE_M_MAX" \
    -e B70_FP8_W8A16_M_MAX="$FP8_W8A16_M_MAX" \
    "${think_env[@]}" \
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
      --attention-backend intel_xpu \
      --linear-attn-backend '$LINEAR_ATTN_BACKEND' \
      --linear-attn-prefill-backend '$LINEAR_ATTN_PREFILL_BACKEND' \
      --mamba-ssm-dtype float32 $grammar_args \
      $graph_args $tool_args \
      --disable-radix-cache --disable-overlap-schedule --skip-server-warmup \
      --chunked-prefill-size '$CHUNKED_PREFILL' \
      --disable-custom-all-reduce --reasoning-parser qwen3 --tp-size '$TP' \
      --context-length '$CTX' --mem-fraction-static '$MEMFRAC' \
      --max-running-requests '$MAXREQ' --host 0.0.0.0 --port '$PORT'" \
    >/dev/null

  local healthy=0
  for i in $(seq 1 180); do
    if ! docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | rg -qx "$NAME"; then
      say "server exited"
      docker logs "$NAME" >"$LOG" 2>&1 || true
      tail -n 160 "$LOG"
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
