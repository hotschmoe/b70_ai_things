#!/usr/bin/env bash
# Qualify the W01 Qwen3.8 W8A8 TP2 long-output control.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/w01_qwen38_w8a8/$STAMP}"
SERVE="$SCRIPT_DIR/serve_qwen38_w8a8.sh"
CORPUS="$SCRIPT_DIR/capture_greedy_corpus.py"
LONG_REPLAY="$SCRIPT_DIR/w01_long_replay.py"
IMG="${IMG:-b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78}"
NAME="w01-qwen38-w8a8-$STAMP"
SERVED="qwen3.8-27b-W8A8-gptq-gdn-rtn-breakable-reclaim500-tp2-bf16kv-w01"
PORT="${PORT:-18080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
HOST_MEM_MIN_GIB="${HOST_MEM_MIN_GIB:-96}"
HOST_SWAP_MAX_GIB="${HOST_SWAP_MAX_GIB:-1}"
CONTAINER_MEMORY_LIMIT_GIB="${CONTAINER_MEMORY_LIMIT_GIB:-64}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-50000}"
WINDOW_TOKENS="${WINDOW_TOKENS:-5000}"
MIN_FINAL_INITIAL_RATIO="${MIN_FINAL_INITIAL_RATIO:-0.80}"
LONG_TIMEOUT="${LONG_TIMEOUT:-5400}"
KERNEL_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "${1:-}" in
  --leased) shift ;;
  "")
    exec env B70_AGENT=w01-qwen38-w8a8 \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

mkdir -p "$RESULT_DIR"

host_snapshot() {
  local available_kib swap_total_kib swap_free_kib
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kib="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
  printf 'utc=%s mem_available_kib=%s swap_used_kib=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$available_kib" \
    "$((swap_total_kib - swap_free_kib))"
}

host_gate() {
  local snapshot available_kib swap_used_kib
  snapshot="$(host_snapshot)"
  echo "$snapshot"
  available_kib="$(sed -n 's/.*mem_available_kib=\([0-9]*\).*/\1/p' <<<"$snapshot")"
  swap_used_kib="$(sed -n 's/.*swap_used_kib=\([0-9]*\).*/\1/p' <<<"$snapshot")"
  [ "$available_kib" -ge $((HOST_MEM_MIN_GIB * 1024 * 1024)) ] &&
    [ "$swap_used_kib" -le $((HOST_SWAP_MAX_GIB * 1024 * 1024)) ]
}

wait_host_gate() {
  local waited=0
  while ! host_gate; do
    if [ "$waited" -ge 600 ]; then
      echo "host resource gate did not recover in 600 seconds" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

host_gate >"$RESULT_DIR/host_preflight.log" || {
  echo "host resource gate failed before GPU work" >&2
  exit 1
}

active=0
gpu_touched=0
monitor_pid=""
server_label=""

serve_env() {
  env \
    IMG="$IMG" NAME="$NAME" PORT="$PORT" SERVED="$SERVED" \
    CTX=65536 MEMFRAC=0.70 MAXREQ=1 MTP=0 DECODE_GRAPH=breakable \
    GRAPH_BS=1 CG_RECLAIM=500 TOOLPARSER=none THINKCAP= \
    HOST_MEM_MIN_GIB="$HOST_MEM_MIN_GIB" \
    HOST_SWAP_MAX_GIB="$HOST_SWAP_MAX_GIB" \
    CONTAINER_MEMORY_LIMIT_GIB="$CONTAINER_MEMORY_LIMIT_GIB" \
    LOG="$RESULT_DIR/server_${server_label}.log" \
    bash "$SERVE" "$@"
}

scan_server_log() {
  local log="$1"
  if rg -i \
    'ZE_RESULT_ERROR_DEVICE_LOST|UR_RESULT_ERROR|linear_stream\.h|GPU (virtual-memory|VM) fault|engine core.*(died|dead)|out of resources|out of memory|Killed process|std::bad_alloc|(^|[^a-z])nan([^a-z]|$)|garbage output' \
    "$log"; then
    echo "fatal server marker in $log" >&2
    return 1
  fi
}

stop_server() {
  local label="$1"
  local rc=0
  local log="$RESULT_DIR/server_${label}.log"
  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/health" \
    >"$RESULT_DIR/endpoint_${label}_before_teardown.txt" 2>&1 || rc=1
  docker logs "$NAME" >"$log" 2>&1 || rc=1
  scan_server_log "$log" || rc=1
  server_label="$label"
  serve_env stop || rc=1
  active=0
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "endpoint remained live after $label teardown" >&2
    rc=1
  fi
  return "$rc"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$active" = 1 ]; then
    stop_server "$server_label" || rc=1
  fi
  if [ "$gpu_touched" = 1 ]; then
    "$REPO/bin/xpu-health" >"$RESULT_DIR/health_post_card.log" 2>&1 || rc=1
    "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
      >"$RESULT_DIR/health_post_collective.log" 2>&1 || rc=1
  fi
  journalctl -k --since "$KERNEL_SINCE" --no-pager \
    >"$RESULT_DIR/kernel_transaction.log" 2>&1 || rc=1
  if rg -i \
    'oom-killer|Out of memory: Killed|GPU (virtual-memory|VM) fault|engine.*(died|dead)|blocked for more than|xe.*(wedg|reset)' \
    "$RESULT_DIR/kernel_transaction.log"; then
    echo "fatal kernel marker" >&2
    rc=1
  fi
  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  echo "VERDICT -> exit=$rc result_dir=$RESULT_DIR"
  exit "$rc"
}
trap cleanup EXIT INT TERM

{
  echo "utc_start=$STAMP"
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "kernel=$(uname -r)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "image=$IMG"
  echo "served=$SERVED"
  echo "tp=2"
  echo "p2p=0"
  echo "target_dtype=bfloat16"
  echo "kv_dtype=bfloat16"
  echo "context=65536"
  echo "mem_fraction=0.70"
  echo "max_running_requests=1"
  echo "decode_graph=breakable"
  echo "graph_bs=1"
  echo "graph_reclaim=500"
  echo "mtp=0"
  echo "radix_cache=off"
  echo "output_tokens=$OUTPUT_TOKENS"
  echo "long_replay_seed=none_native_greedy"
  echo "window_tokens=$WINDOW_TOKENS"
  echo "minimum_final_initial_ratio=$MIN_FINAL_INITIAL_RATIO"
  echo "host_mem_min_gib=$HOST_MEM_MIN_GIB"
  echo "host_swap_max_gib=$HOST_SWAP_MAX_GIB"
  echo "container_memory_limit_gib=$CONTAINER_MEMORY_LIMIT_GIB"
  sha256sum "$SERVE" "$CORPUS" "$LONG_REPLAY" \
    "$REPO/sglang/refresh/b70_xpu_w8a8.py"
  dpkg-query -W -f='${Package}=${Version}\n' 2>/dev/null | \
    rg '^(intel-level-zero-gpu|intel-opencl-icd|libze1|intel-oneapi-runtime-ccl)=' || true
} >"$RESULT_DIR/config.txt"

(
  while :; do
    host_snapshot
    awk '{print "memory_psi " $0}' /proc/pressure/memory
    sleep 5
  done
) >"$RESULT_DIR/host_memory_monitor.log" &
monitor_pid=$!

gpu_touched=1
"$REPO/bin/xpu-health" 2>&1 | tee "$RESULT_DIR/health_pre_card.log"
"$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
  | tee "$RESULT_DIR/health_pre_collective.log"

start_server() {
  local label="$1" expected_bytes
  server_label="$label"
  active=1
  serve_env start | tee "$RESULT_DIR/start_${label}.log"
  docker logs "$NAME" >"$RESULT_DIR/server_${label}.log" 2>&1
  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
    >"$RESULT_DIR/models_${label}.json"
  jq -e --arg model "$SERVED" '[.data[].id] == [$model]' \
    "$RESULT_DIR/models_${label}.json" >/dev/null
  docker inspect "$NAME" --format \
    'image={{.Image}} memory={{.HostConfig.Memory}} memory_swap={{.HostConfig.MemorySwap}} oom_score_adj={{.HostConfig.OomScoreAdj}}' \
    >"$RESULT_DIR/container_resources_${label}.txt"
  expected_bytes=$((CONTAINER_MEMORY_LIMIT_GIB * 1024 * 1024 * 1024))
  rg -q "memory=$expected_bytes memory_swap=$expected_bytes oom_score_adj=500" \
    "$RESULT_DIR/container_resources_${label}.txt"
  docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sort >"$RESULT_DIR/container_env_${label}.txt"
  rg -q '^CCL_TOPO_P2P_ACCESS=0$' "$RESULT_DIR/container_env_${label}.txt"
  rg -q '^B70_XPU_CG_RECLAIM=500$' "$RESULT_DIR/container_env_${label}.txt"
  rg -q "dtype='bfloat16'.*kv_cache_dtype='bfloat16'" \
    "$RESULT_DIR/server_${label}.log"
  [ "$(rg -c '\[b70-xpu-graph\] reclaim enabled:' "$RESULT_DIR/server_${label}.log")" -ge 2 ]
  docker exec -i "$NAME" python - <<'PY' >"$RESULT_DIR/runtime_${label}.json"
import importlib.metadata
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "sglang": importlib.metadata.version("sglang"),
}, sort_keys=True))
PY
  docker exec "$NAME" bash -c \
    'sha256sum /opt/venv/lib/libccl.so.1.0 /opt/venv/lib/ccl/kernels/* 2>/dev/null' \
    >"$RESULT_DIR/runtime_hashes_${label}.txt"
}

start_server a
python3 "$CORPUS" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --output "$RESULT_DIR/corpus_server_a.json"
stop_server a

"$REPO/bin/xpu-health" 2>&1 | tee "$RESULT_DIR/health_inter_card.log"
"$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
  | tee "$RESULT_DIR/health_inter_collective.log"
wait_host_gate | tee "$RESULT_DIR/host_before_server_b.log"

start_server b
python3 "$CORPUS" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --reference "$RESULT_DIR/corpus_server_a.json" \
  --output "$RESULT_DIR/corpus_server_b.json"

python3 "$LONG_REPLAY" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --output-tokens "$OUTPUT_TOKENS" --window-tokens "$WINDOW_TOKENS" \
  --minimum-final-initial-ratio "$MIN_FINAL_INITIAL_RATIO" \
  --timeout "$LONG_TIMEOUT" --json-out "$RESULT_DIR/long_replay.json" \
  | tee "$RESULT_DIR/long_replay.log"

docker logs "$NAME" >"$RESULT_DIR/server_b.log" 2>&1
[ "$(rg -c '\[b70-xpu-graph\] executable re-instantiated' "$RESULT_DIR/server_b.log")" -ge 2 ]
scan_server_log "$RESULT_DIR/server_b.log"
echo "RESULT -> W01 passed workload gates; teardown and post-health pending"
