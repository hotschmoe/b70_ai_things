#!/usr/bin/env bash
# Reproduce Steve Seguin's public R50 graph-off MTP1 result with local safety gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260901/source}"
SOURCE_COMMIT="${SOURCE_COMMIT:-6adab048f80c4f1161fb812e0387b124a9624494}"
PACKAGE="$SOURCE/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70"
LAUNCHER="$PACKAGE/run-w8a16-mtp1-strict-server.sh"
BENCH="$PACKAGE/bench-w8a16-mtp1-strict.sh"
IMAGE="${IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50-s01}"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/qwen38_fp8_steve_r50_strict/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/qwen38_fp8_steve_r50_strict/$STAMP}"
NAME="${NAME:-qwen38-fp8-steve-r50-strict-$STAMP}"
PORT="${PORT:-18124}"
SERVED="${SERVED:-qwen38-fp8-block-w8a16-mtp1}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
READY_STALL="${READY_STALL:-360}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-64}"

case "${1:-}" in
  --leased) shift ;;
  --print-config)
    printf '%s\n' \
      "source=$SOURCE" \
      "source_commit=$SOURCE_COMMIT" \
      "image=$IMAGE" \
      "model_dir=$MODEL_DIR" \
      "result_dir=$RESULT_DIR" \
      "cache_dir=$CACHE_DIR" \
      "container=$NAME" \
      "port=$PORT" \
      "served_model=$SERVED" \
      "profile=Steve R50 strict graph-off MTP1 P2P1 cache-off c1" \
      "max_model_len=1024" \
      "max_num_seqs=1" \
      "max_num_batched_tokens=1024"
    exit 0
    ;;
  '')
    exec env B70_AGENT=qwen38-fp8-steve-r50-strict \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) printf 'usage: %s [--print-config]\n' "$0" >&2; exit 2 ;;
esac

for pair in "PORT:$PORT" "READY_TIMEOUT:$READY_TIMEOUT" \
  "READY_STALL:$READY_STALL" "HEALTH_TIMEOUT:$HEALTH_TIMEOUT" \
  "MIN_AVAILABLE_GIB:$MIN_AVAILABLE_GIB"; do
  value="${pair#*:}"
  case "$value" in
    ''|*[!0-9]*|0) printf '%s must be positive\n' "${pair%%:*}" >&2; exit 2 ;;
  esac
done

[ ! -e "$RESULT_DIR" ] || { printf 'RESULT_DIR must be new: %s\n' "$RESULT_DIR" >&2; exit 1; }
[ ! -e "$CACHE_DIR" ] || { printf 'CACHE_DIR must be new: %s\n' "$CACHE_DIR" >&2; exit 1; }
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  printf 'source checkout is not at %s\n' "$SOURCE_COMMIT" >&2
  exit 1
}
for required in "$MODEL_DIR" "$LAUNCHER" "$BENCH"; do
  [ -e "$required" ] || { printf 'missing input: %s\n' "$required" >&2; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null

mkdir -p "$RESULT_DIR"
journal_start="$(date +%s)"
swap_start_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
server_pid=""
monitor_pid=""

stop_server() {
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ -n "$server_pid" ]; then wait "$server_pid" 2>/dev/null || true; fi
  server_pid=""
}

stop_monitor() {
  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
}

run_health() {
  local label="$1"
  "$REPO/bin/xpu-health" --img "$IMAGE" \
    2>&1 | tee "$RESULT_DIR/${label}-card-health.log"
  env IMG="$IMAGE" "$REPO/bin/xpu-collective-health" \
    --p2p 0 --timeout "$HEALTH_TIMEOUT" \
    2>&1 | tee "$RESULT_DIR/${label}-collective-health.log"
}

cleanup() {
  local rc=$?
  set +e
  stop_server
  stop_monitor
  journalctl -k --since "@$journal_start" --no-pager \
    >"$RESULT_DIR/kernel-journal.log" 2>"$RESULT_DIR/kernel-journal.err"
  if [ "$rc" -ne 0 ]; then
    "$REPO/bin/xe-reset" --method rebind >"$RESULT_DIR/recovery.log" 2>&1 || rc=1
    run_health failure-post >"$RESULT_DIR/failure-post.log" 2>&1 || rc=1
  fi
  printf '%s\n' "$rc" >"$RESULT_DIR/qualifier.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

host_gate() {
  local available swap_used
  available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_used="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
  [ "$available" -ge "$((MIN_AVAILABLE_GIB * 1024 * 1024))" ] || {
    printf 'host gate failed: MemAvailable below %s GiB\n' "$MIN_AVAILABLE_GIB" >&2
    return 1
  }
  [ "$swap_used" -le 1048576 ] || {
    printf 'host gate failed: swap use above 1 GiB\n' >&2
    return 1
  }
}

monitor_host() {
  local available swap_total swap_free
  while :; do
    available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_total="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    printf 'mem_available_kib=%s swap_used_kib=%s\n' \
      "$available" "$((swap_total - swap_free))"
    cat /proc/pressure/cpu /proc/pressure/io /proc/pressure/memory
    docker stats --no-stream --format \
      'docker name={{.Name}} mem={{.MemUsage}} mem_percent={{.MemPerc}} pids={{.PIDs}}' \
      "$NAME" 2>/dev/null || true
    sleep 5
  done >"$RESULT_DIR/host-monitor.log" 2>&1
}

wait_ready() {
  local start now signature last_signature last_progress
  start="$(date +%s)"
  last_progress="$start"
  last_signature=""
  while :; do
    curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$server_pid" 2>/dev/null || {
      printf 'server exited before readiness\n' >&2
      tail -120 "$RESULT_DIR/server.log" >&2 || true
      return 1
    }
    now="$(date +%s)"
    signature="$(sha256sum "$RESULT_DIR/server.log" | awk '{print $1}')"
    if [ "$signature" != "$last_signature" ]; then
      last_signature="$signature"
      last_progress="$now"
    fi
    [ "$((now - last_progress))" -lt "$READY_STALL" ] || {
      printf 'server readiness stalled\n' >&2
      return 1
    }
    [ "$((now - start))" -lt "$READY_TIMEOUT" ] || {
      printf 'server readiness timeout\n' >&2
      return 1
    }
    sleep 5
  done
}

echo "CONFIG"
env SOURCE="$SOURCE" SOURCE_COMMIT="$SOURCE_COMMIT" IMAGE="$IMAGE" \
  MODEL_DIR="$MODEL_DIR" RESULT_DIR="$RESULT_DIR" CACHE_DIR="$CACHE_DIR" \
  NAME="$NAME" PORT="$PORT" SERVED="$SERVED" "$0" --print-config \
  | tee "$RESULT_DIR/config.txt"
echo "COMMAND -> bin/gpu-run bash vllm/fp8/qualify_qwen38_fp8_steve_r50_strict.sh --leased" \
  | tee "$RESULT_DIR/command.txt"
host_gate
"$PACKAGE/verify-image-contract.sh" mtp1-serial-fa-split-gdn "$IMAGE" \
  | tee "$RESULT_DIR/image-contract.txt"
docker image inspect "$IMAGE" >"$RESULT_DIR/image-inspect.json"
run_health pre

monitor_host &
monitor_pid=$!
env IMAGE="$IMAGE" MODEL_DIR="$MODEL_DIR" VLLM_CACHE_DIR="$CACHE_DIR" \
  CONTAINER_NAME="$NAME" PORT="$PORT" SERVED_MODEL_NAME="$SERVED" \
  MAX_MODEL_LEN=1024 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=1024 \
  bash "$LAUNCHER" >"$RESULT_DIR/server.log" 2>&1 &
server_pid=$!
wait_ready

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$RESULT_DIR/models.json"
python3 - "$RESULT_DIR/models.json" "$SERVED" <<'PY'
import json
import sys

ids = [item["id"] for item in json.load(open(sys.argv[1], encoding="ascii"))["data"]]
assert ids == [sys.argv[2]], ids
PY
docker inspect "$NAME" >"$RESULT_DIR/container-inspect.json"
env OUT_DIR="$RESULT_DIR/attempt-1" BASE_URL="http://127.0.0.1:$PORT" \
  MODEL_NAME="$SERVED" PROFILE_LABEL="mtp1-r50-strict-local" \
  ATTEMPT_LABEL="local-fresh-attempt-1" bash "$BENCH" \
  | tee "$RESULT_DIR/attempt-1.stdout"

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/health" \
  >"$RESULT_DIR/endpoint-post-health.json"
docker stats --no-stream --format '{{json .}}' "$NAME" >"$RESULT_DIR/docker-stats.json"
docker logs "$NAME" >"$RESULT_DIR/server-final.log" 2>&1 || true
stop_server
stop_monitor
run_health post
journalctl -k --since "@$journal_start" --no-pager >"$RESULT_DIR/kernel-journal.log"

python3 - "$RESULT_DIR/attempt-1/performance.json" "$RESULT_DIR/verdict.json" <<'PY'
import json
import pathlib
import sys

performance = json.load(open(sys.argv[1], encoding="ascii"))
rate = performance["summary"]["class_balanced_tok_s_1_100_intervals_after_ttft"]["median"]
reference = 51.808087
floor = reference * 0.9
verdict = {
    "schema": "b70.qwen38-fp8-steve-r50-strict.v1",
    "local_tok_s": rate,
    "steve_reference_tok_s": reference,
    "within_10_percent_floor_tok_s": floor,
    "within_10_percent": rate >= floor,
    "strict_workload_gate": performance["realistic_final_gate"]["passed"],
    "cached_tokens_all_zero": performance["fresh_response_validity"]["cached_tokens_all_zero"],
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
print(json.dumps(verdict, indent=2, sort_keys=True))
assert verdict["within_10_percent"]
assert verdict["strict_workload_gate"]
assert verdict["cached_tokens_all_zero"]
PY

if grep -Eqi 'xe 0000:(43|47):00\.0.*(reset|fault|timeout|timed out|fatal|wedged|failed)' \
  "$RESULT_DIR/kernel-journal.log"; then
  printf 'new GPU kernel fault event detected\n' >&2
  exit 1
fi
max_swap_kib="$(awk -F= '/swap_used_kib=/ {print $3}' "$RESULT_DIR/host-monitor.log" | sort -n | tail -1)"
max_swap_kib="${max_swap_kib:-0}"
[ "$max_swap_kib" -le "$swap_start_kib" ] || {
  printf 'strict reproduction increased host swap: baseline_kib=%s max_kib=%s\n' \
    "$swap_start_kib" "$max_swap_kib" >&2
  exit 1
}
echo "RESULT"
cat "$RESULT_DIR/verdict.json"
echo "VERDICT -> strict R50 reproduction passed workload, health, teardown, and 10 percent speed gates"
printf '0\n' >"$RESULT_DIR/qualifier.rc"
trap - EXIT INT TERM HUP
