#!/usr/bin/env bash
# Experimental long-context, cache-on daily-driver shape for Steve's R187
# Qwen3.8 FP8 MTP5 XPU-graph profile. The foreground process holds both leases.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260904-source-r187}"
SOURCE_COMMIT="${SOURCE_COMMIT:-8319e0964df12a1f0bc920301efc662ac49a949e}"
PACKAGE="$SOURCE/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70"
LAUNCHER="$PACKAGE/run-w8a16-mtp1-strict-server.sh"
IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-gdn-split-mixed-r156}"
IMAGE_ID="${IMAGE_ID:-sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152}"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/qwen38_fp8_steve_mtp5_daily_r187}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/qwen38_fp8_steve_mtp5_daily_r187/$STAMP}"
NAME="${NAME:-qwen38-fp8-steve-mtp5-daily-r187}"
SERVED="${SERVED:-qwen3.8-27b-FP8-official-W8A16-mtp5-r187-xpugraph-cacheon-ctx237568-daily}"
PORT="${PORT:-18080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-237568}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],"max_cudagraph_capture_size":24,"splitting_ops":[],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"deterministic":true,"triton.autotune_pointwise":false,"benchmark_epilogue_fusion":false}}'
EXTRA_SERVE_ARGS='--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":true,"reasoning_effort":"xhigh"}'

print_config() {
  printf '%s\n' \
    "source=$SOURCE" \
    "source_commit=$SOURCE_COMMIT" \
    "image=$IMAGE" \
    "image_id=$IMAGE_ID" \
    "model_dir=$MODEL_DIR" \
    "cache_dir=$CACHE_DIR" \
    "result_dir=$RESULT_DIR" \
    "container=$NAME" \
    "served_model=$SERVED" \
    "listen=127.0.0.1:$PORT" \
    "max_model_len=$MAX_MODEL_LEN" \
    "max_num_seqs=$MAX_NUM_SEQS" \
    "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS" \
    "prefix_cache=align" \
    "mtp=5" \
    "xpu_graph=1" \
    "graph_capture_sizes=1..24"
}

run_health() {
  local label=$1
  "$REPO/bin/xpu-health" --img "$IMAGE" 2>&1 \
    | tee "$RESULT_DIR/${label}-card-health.log"
  env IMG="$IMAGE" "$REPO/bin/xpu-collective-health" \
    --p2p 0 --timeout "$HEALTH_TIMEOUT" 2>&1 \
    | tee "$RESULT_DIR/${label}-collective-health.log"
}

case "${1:-start}" in
  --print-config)
    print_config
    exit 0
    ;;
  status)
    docker ps --filter "name=^/${NAME}$" --format '{{.Names}} {{.Status}} {{.Ports}}'
    exit 0
    ;;
  stop)
    docker stop -t 60 "$NAME"
    exit 0
    ;;
  start)
    exec env B70_AGENT=qwen38-fp8-steve-mtp5-daily-r187 \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  --leased)
    ;;
  *)
    printf 'usage: %s start|stop|status|--print-config\n' "$0" >&2
    exit 2
    ;;
esac

for pair in "PORT:$PORT" "MAX_MODEL_LEN:$MAX_MODEL_LEN" \
  "MAX_NUM_SEQS:$MAX_NUM_SEQS" \
  "MAX_NUM_BATCHED_TOKENS:$MAX_NUM_BATCHED_TOKENS" \
  "HEALTH_TIMEOUT:$HEALTH_TIMEOUT"; do
  value="${pair#*:}"
  case "$value" in
    ''|*[!0-9]*|0) printf '%s must be positive\n' "${pair%%:*}" >&2; exit 2 ;;
  esac
done

[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  printf 'source checkout is not at %s\n' "$SOURCE_COMMIT" >&2
  exit 1
}
[[ -e "$MODEL_DIR" && -x "$LAUNCHER" ]] || {
  printf 'model or launcher input is missing\n' >&2
  exit 1
}
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] || {
  printf 'image ID mismatch\n' >&2
  exit 1
}
! docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME" || {
  printf 'container already exists: %s\n' "$NAME" >&2
  exit 1
}
[[ ! -e "$RESULT_DIR" ]] || {
  printf 'result directory already exists: %s\n' "$RESULT_DIR" >&2
  exit 1
}

mkdir -p "$RESULT_DIR" "$CACHE_DIR"
journal_start="$(date +%s)"

cleanup() {
  local rc=$? health_rc
  set +e
  docker rm -f "$NAME" >/dev/null 2>&1
  journalctl -k --since "@$journal_start" --no-pager \
    >"$RESULT_DIR/kernel-journal.log" 2>"$RESULT_DIR/kernel-journal.err"
  if [[ "$rc" -ne 0 ]]; then
    "$REPO/bin/xe-reset" --method rebind >"$RESULT_DIR/recovery.log" 2>&1
  fi
  run_health post >"$RESULT_DIR/post-health.stdout" 2>&1
  health_rc=$?
  if [[ "$rc" -eq 0 && "$health_rc" -ne 0 ]]; then
    rc=$health_rc
  fi
  printf '%s\n' "$rc" >"$RESULT_DIR/server.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

print_config | tee "$RESULT_DIR/config.txt"
printf 'COMMAND -> bin/gpu-run bash %s --leased\n' "$0" \
  | tee "$RESULT_DIR/command.txt"
env \
  EXPECTED_XPU_EXTENSION_SHA256=f912e12de1d79206221142c9a50af2aba70d2c77c735c9cd2d5d8d9def0740d1 \
  EXPECTED_XPU_OPS_SHA256=6a7761930cd8b9e3f67902648ba5aaaf708567cebf70fcedda595d698f26b064 \
  "$PACKAGE/verify-image-contract.sh" mtp1-serial-fa-split-gdn "$IMAGE" \
  | tee "$RESULT_DIR/image-contract.txt"
run_health pre

env \
  IMAGE="$IMAGE" EXPECTED_IMAGE_ID="$IMAGE_ID" \
  MODEL_DIR="$MODEL_DIR" VLLM_CACHE_DIR="$CACHE_DIR" \
  CONTAINER_NAME="$NAME" PORT="$PORT" SERVED_MODEL_NAME="$SERVED" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  GPU_MEMORY_UTILIZATION=0.96 \
  CONTAINER_MEMORY=32g CONTAINER_MEMORY_SWAP=32g \
  EXPECTED_XPU_EXTENSION_SHA256=f912e12de1d79206221142c9a50af2aba70d2c77c735c9cd2d5d8d9def0740d1 \
  EXPECTED_XPU_OPS_SHA256=6a7761930cd8b9e3f67902648ba5aaaf708567cebf70fcedda595d698f26b064 \
  VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  VLLM_XPU_GDN_SPLIT_MIXED=1 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4=1 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE=128 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_SCALE_DTYPE=bf16 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_CHUNK_ROWS=2048 \
  SPECULATIVE_CONFIG='{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
  COMPILATION_CONFIG="$COMPILATION_CONFIG" \
  EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS" \
  bash "$LAUNCHER" 2>&1 | tee "$RESULT_DIR/server.log"
