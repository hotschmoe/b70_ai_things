#!/usr/bin/env bash
# Run the matched W02 eager, breakable, and reclaim500 comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/w02_qwen38_w8a8/$STAMP}"
SERVE="$SCRIPT_DIR/serve_qwen38_w8a8.sh"
CORPUS="$SCRIPT_DIR/capture_greedy_corpus.py"
REPLAY="$SCRIPT_DIR/w01_long_replay.py"
COMPARE="$SCRIPT_DIR/w02_compare.py"
IMG="${IMG:-b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78}"
NAME="w02-qwen38-w8a8-$STAMP"
SERVED="qwen3.8-27b-W8A8-gptq-gdn-rtn-tp2-bf16kv-w02"
PORT="${PORT:-18080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
HOST_MEM_MIN_GIB="${HOST_MEM_MIN_GIB:-96}"
HOST_SWAP_MAX_GIB="${HOST_SWAP_MAX_GIB:-1}"
CONTAINER_MEMORY_LIMIT_GIB="${CONTAINER_MEMORY_LIMIT_GIB:-64}"
WARMUP_TOKENS="${WARMUP_TOKENS:-768}"
MEASURED_TOKENS="${MEASURED_TOKENS:-2048}"
MEASURED_REPEATS="${MEASURED_REPEATS:-3}"
KERNEL_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "${1:-}" in
  --leased) shift ;;
  "")
    exec env B70_AGENT=w02-qwen38-w8a8 \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

[ "$MEASURED_REPEATS" -ge 2 ] 2>/dev/null || {
  echo "MEASURED_REPEATS must be at least 2" >&2
  exit 2
}

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
current_arm=""
current_graph=""
current_reclaim=""

serve_env() {
  env \
    IMG="$IMG" NAME="$NAME" PORT="$PORT" SERVED="$SERVED" \
    CTX=65536 MEMFRAC=0.70 MAXREQ=1 MTP=0 \
    DECODE_GRAPH="$current_graph" GRAPH_BS=1 CG_RECLAIM="$current_reclaim" \
    TOOLPARSER=none THINKCAP= \
    HOST_MEM_MIN_GIB="$HOST_MEM_MIN_GIB" \
    HOST_SWAP_MAX_GIB="$HOST_SWAP_MAX_GIB" \
    CONTAINER_MEMORY_LIMIT_GIB="$CONTAINER_MEMORY_LIMIT_GIB" \
    LOG="$RESULT_DIR/$current_arm/server.log" \
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

stop_arm() {
  local arm="$current_arm"
  local rc=0
  local log="$RESULT_DIR/$arm/server.log"
  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/health" \
    >"$RESULT_DIR/$arm/endpoint_before_teardown.txt" 2>&1 || rc=1
  docker logs "$NAME" >"$log" 2>&1 || rc=1
  scan_server_log "$log" || rc=1
  serve_env stop || rc=1
  active=0
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "endpoint remained live after $arm teardown" >&2
    rc=1
  fi
  return "$rc"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$active" = 1 ]; then
    stop_arm || rc=1
  fi
  if [ "$gpu_touched" = 1 ]; then
    "$REPO/bin/xpu-health" >"$RESULT_DIR/health_final_card.log" 2>&1 || rc=1
    "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
      >"$RESULT_DIR/health_final_collective.log" 2>&1 || rc=1
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
  echo "arms=eager,breakable,reclaim500"
  echo "tp=2"
  echo "p2p=0"
  echo "target_dtype=bfloat16"
  echo "kv_dtype=bfloat16"
  echo "context=65536"
  echo "mem_fraction=0.70"
  echo "max_running_requests=1"
  echo "mtp=0"
  echo "radix_cache=off"
  echo "think_cap=empty"
  echo "warmup_tokens=$WARMUP_TOKENS"
  echo "measured_tokens=$MEASURED_TOKENS"
  echo "measured_repeats=$MEASURED_REPEATS"
  echo "stream_interval=128"
  echo "host_mem_min_gib=$HOST_MEM_MIN_GIB"
  echo "host_swap_max_gib=$HOST_SWAP_MAX_GIB"
  echo "container_memory_limit_gib=$CONTAINER_MEMORY_LIMIT_GIB"
  sha256sum "$SERVE" "$CORPUS" "$REPLAY" "$COMPARE" \
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
  2>&1 | tee "$RESULT_DIR/health_pre_collective.log"

start_arm() {
  local arm="$1"
  local graph="$2"
  local reclaim="$3"
  local arm_dir="$RESULT_DIR/$arm"
  local expected_bytes
  current_arm="$arm"
  current_graph="$graph"
  current_reclaim="$reclaim"
  mkdir -p "$arm_dir"
  active=1
  serve_env start | tee "$arm_dir/start.log"
  docker logs "$NAME" >"$arm_dir/server.log" 2>&1
  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
    >"$arm_dir/models.json"
  jq -e --arg model "$SERVED" '[.data[].id] == [$model]' \
    "$arm_dir/models.json" >/dev/null
  docker inspect "$NAME" --format \
    'image={{.Image}} memory={{.HostConfig.Memory}} memory_swap={{.HostConfig.MemorySwap}} oom_score_adj={{.HostConfig.OomScoreAdj}}' \
    >"$arm_dir/container_resources.txt"
  expected_bytes=$((CONTAINER_MEMORY_LIMIT_GIB * 1024 * 1024 * 1024))
  rg -q "memory=$expected_bytes memory_swap=$expected_bytes oom_score_adj=500" \
    "$arm_dir/container_resources.txt"
  docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sort >"$arm_dir/container_env.txt"
  rg -q '^CCL_TOPO_P2P_ACCESS=0$' "$arm_dir/container_env.txt"
  rg -q "^B70_XPU_CG_RECLAIM=$reclaim$" "$arm_dir/container_env.txt"
  if [ "$graph" = breakable ]; then
    rg -q '^B70_XPU_BREAKABLE_GRAPH=1$' "$arm_dir/container_env.txt"
    [ "$(rg -c '\[b70-xpu-graph\] enabled breakable decode' "$arm_dir/server.log")" -ge 2 ]
  else
    rg -q '^B70_XPU_BREAKABLE_GRAPH=0$' "$arm_dir/container_env.txt"
    ! rg -q '\[b70-xpu-graph\] enabled breakable decode' "$arm_dir/server.log"
  fi
  rg -q "dtype='bfloat16'.*kv_cache_dtype='bfloat16'" "$arm_dir/server.log"
  docker exec -i "$NAME" python - <<'PY' >"$arm_dir/runtime.txt"
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
    >"$arm_dir/runtime_hashes.txt"
}

run_arm() {
  local arm="$1"
  local graph="$2"
  local reclaim="$3"
  local arm_dir="$RESULT_DIR/$arm"
  local reference_args=()
  local repeat

  start_arm "$arm" "$graph" "$reclaim"
  if [ "$arm" != eager ]; then
    reference_args=(--reference "$RESULT_DIR/eager/corpus.json")
  fi
  python3 "$CORPUS" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    "${reference_args[@]}" --output "$arm_dir/corpus.json"
  python3 "$REPLAY" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --output-tokens "$WARMUP_TOKENS" --window-tokens 256 \
    --stream-interval 128 \
    --timeout 900 --json-out "$arm_dir/warmup.json" \
    | tee "$arm_dir/warmup.log"
  for repeat in $(seq 1 "$MEASURED_REPEATS"); do
    python3 "$REPLAY" --base "http://127.0.0.1:$PORT" --model "$SERVED" \
      --output-tokens "$MEASURED_TOKENS" --window-tokens 512 \
      --stream-interval 128 \
      --timeout 900 --json-out "$arm_dir/measured_${repeat}.json" \
      | tee "$arm_dir/measured_${repeat}.log"
  done
  [ "$(jq -r '.output_ids_sha256' "$arm_dir"/measured_*.json | sort -u | wc -l)" -eq 1 ]
  [ "$(jq -r '.text_sha256' "$arm_dir"/measured_*.json | sort -u | wc -l)" -eq 1 ]
  docker logs "$NAME" >"$arm_dir/server.log" 2>&1
  if [ "$reclaim" -gt 0 ]; then
    [ "$(rg -c '\[b70-xpu-graph\] executable re-instantiated' "$arm_dir/server.log")" -ge 2 ]
  else
    ! rg -q '\[b70-xpu-graph\] executable re-instantiated' "$arm_dir/server.log"
  fi
  scan_server_log "$arm_dir/server.log"
  stop_arm
  "$REPO/bin/xpu-health" >"$arm_dir/health_after_card.log" 2>&1
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
    >"$arm_dir/health_after_collective.log" 2>&1
  wait_host_gate >"$arm_dir/host_after.log"
}

run_arm eager 0 0
run_arm breakable breakable 0
run_arm reclaim500 breakable 500

python3 "$COMPARE" --root "$RESULT_DIR" --repeats "$MEASURED_REPEATS" \
  --json-out "$RESULT_DIR/comparison.json" | tee "$RESULT_DIR/comparison.log"
echo "RESULT -> W02 short mechanism comparison passed; long no-reclaim canary pending"
