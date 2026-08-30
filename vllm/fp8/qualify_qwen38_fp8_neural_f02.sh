#!/usr/bin/env bash
# Two-fresh-server F02 qualification of the Neural.Download Qwen3.8 FP8 MTP0
# path under the local P2P-off and no-swap safety policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source}"
SOURCE_COMMIT="0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
MODEL_MANIFEST="$SOURCE/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/model-direct.json"
MODEL_VERIFY="$SOURCE/repro/qwen38-27b-autoround-int4-b70/scripts/verify-model-direct.py"
SUITE="$SOURCE/repro/qwen36-27b-autoround-int4-b70/realistic-suite-v1.json"
SUITE_SHA256="df03f49d36c36d2b8ac4cd117b7cb2e42c74878af1f6926690ebb89eeccd47ac"
BENCH="$SOURCE/scripts/bench-openai-realistic-suite.py"
CANARIES="$SOURCE/scripts/neural-download-canaries.py"
PUBLISHER_A="${PUBLISHER_A:-$SOURCE/experiments/qwen38-27b-b70/data/qwen38-fp8-deterministic-compiled-workwait-20260828-r15a/performance.json}"
PUBLISHER_B="${PUBLISHER_B:-$SOURCE/experiments/qwen38-27b-b70/data/qwen38-fp8-deterministic-compiled-workwait-20260828-r15b/performance.json}"
LAUNCHER="$SCRIPT_DIR/serve_qwen38_fp8_neural_f02.sh"
CAMPAIGN_ID="${CAMPAIGN_ID:-f02}"
CAMPAIGN_LABEL="${CAMPAIGN_LABEL:-F02}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-qwen38-fp8-neural-f02}"
ANALYZER_SCHEMA="${ANALYZER_SCHEMA:-b70.qwen38-fp8-neural-f02.v2}"
COMPLETION_ROUTE="${COMPLETION_ROUTE:-explicit-work-wait}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/f02_qwen38_fp8_neural/$STAMP}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f02_qwen38_fp8_neural/$STAMP}"
SERVED="${SERVED:-qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02}"
PORT="${PORT:-18187}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
READY_STALL="${READY_STALL:-360}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-96}"
MAX_SWAP_USED_MIB="${MAX_SWAP_USED_MIB:-1024}"
ATTEMPTS="${ATTEMPTS:-2}"
SHARED_CACHE="${SHARED_CACHE:-0}"
REQUIRE_REFERENCE_EXACT="${REQUIRE_REFERENCE_EXACT:-0}"

case "${1:-}" in
  --leased) shift ;;
  --print-config)
    echo "source=$SOURCE"
    echo "source_commit=$SOURCE_COMMIT"
    echo "model_dir=$MODEL_DIR"
    echo "suite=$SUITE"
    echo "result_dir=$RESULT_DIR"
    echo "cache_root=$CACHE_ROOT"
    echo "attempts=$ATTEMPTS"
    echo "campaign_id=$CAMPAIGN_ID"
    echo "completion_route=$COMPLETION_ROUTE"
    echo "shared_cache=$SHARED_CACHE"
    echo "require_reference_exact=$REQUIRE_REFERENCE_EXACT"
    echo "container_prefix=$CONTAINER_PREFIX"
    echo "p2p=0"
    echo "swap_extra=0"
    env NAME="${CONTAINER_PREFIX}-${STAMP}-attempt-N" \
      ALLOW_EXISTING_CACHE="$SHARED_CACHE" "$LAUNCHER" --print-config
    exit 0
    ;;
  '')
    exec env B70_AGENT="${CAMPAIGN_ID}-qwen38-fp8-neural" \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0 [--print-config]" >&2; exit 2 ;;
esac

for pair in \
  "PORT:$PORT" "READY_TIMEOUT:$READY_TIMEOUT" "READY_STALL:$READY_STALL" \
  "HEALTH_TIMEOUT:$HEALTH_TIMEOUT" "MIN_AVAILABLE_GIB:$MIN_AVAILABLE_GIB" \
  "MAX_SWAP_USED_MIB:$MAX_SWAP_USED_MIB" "ATTEMPTS:$ATTEMPTS"; do
  value="${pair#*:}"
  case "$value" in ''|*[!0-9]*|0) echo "${pair%%:*} must be positive" >&2; exit 2 ;; esac
done
[ "$ATTEMPTS" -ge 2 ] || { echo "ATTEMPTS must be at least 2" >&2; exit 2; }
case "$SHARED_CACHE" in
  0|1) ;;
  *) echo "SHARED_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
case "$REQUIRE_REFERENCE_EXACT" in
  0|1) ;;
  *) echo "REQUIRE_REFERENCE_EXACT must be 0 or 1" >&2; exit 2 ;;
esac
[ ! -e "$RESULT_DIR" ] || { echo "RESULT_DIR must be new: $RESULT_DIR" >&2; exit 1; }
[ ! -e "$CACHE_ROOT" ] || { echo "CACHE_ROOT must be new: $CACHE_ROOT" >&2; exit 1; }
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  echo "source checkout is not at $SOURCE_COMMIT" >&2
  exit 1
}
for required in "$MODEL_DIR" "$MODEL_MANIFEST" "$MODEL_VERIFY" "$SUITE" \
  "$BENCH" "$CANARIES" "$PUBLISHER_A" "$PUBLISHER_B" "$LAUNCHER"; do
  [ -e "$required" ] || { echo "missing input: $required" >&2; exit 1; }
done
[ "$(sha256sum "$SUITE" | awk '{print $1}')" = "$SUITE_SHA256" ] || {
  echo "suite hash mismatch" >&2
  exit 1
}

mkdir -p "$RESULT_DIR" "$CACHE_ROOT"
journal_start="$(date +%s)"
current_name=""
server_pid=""
monitor_pid=""

stop_server() {
  if [ -n "$current_name" ]; then
    docker stop -t 30 "$current_name" >/dev/null 2>&1 || true
    docker rm -f "$current_name" >/dev/null 2>&1 || true
  fi
  if [ -n "$server_pid" ]; then
    wait "$server_pid" 2>/dev/null || true
  fi
  current_name=""
  server_pid=""
}

stop_monitor() {
  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
}

cleanup() {
  local rc=$?
  set +e
  stop_server
  stop_monitor
  journalctl -k --since "@${journal_start}" --no-pager \
    >"$RESULT_DIR/kernel-journal.log" 2>"$RESULT_DIR/kernel-journal.err"
  echo "$rc" >"$RESULT_DIR/qualifier.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

memory_snapshot() {
  local label="$1" mem_available_kib swap_total_kib swap_free_kib
  mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  printf '%s epoch=%s mem_available_kib=%s swap_used_kib=%s\n' \
    "$label" "$(date +%s)" "$mem_available_kib" \
    "$((swap_total_kib - swap_free_kib))"
}

host_gate() {
  local mem_available_kib swap_total_kib swap_free_kib
  mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  [ "$mem_available_kib" -ge "$((MIN_AVAILABLE_GIB * 1024 * 1024))" ] || {
    echo "host gate failed: MemAvailable below ${MIN_AVAILABLE_GIB} GiB" >&2
    return 1
  }
  [ "$((swap_total_kib - swap_free_kib))" -le "$((MAX_SWAP_USED_MIB * 1024))" ] || {
    echo "host gate failed: swap use above ${MAX_SWAP_USED_MIB} MiB" >&2
    return 1
  }
}

run_health() {
  local label="$1"
  "$REPO/bin/xpu-health" 2>&1 | tee "$RESULT_DIR/${label}-card-health.log"
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout "$HEALTH_TIMEOUT" \
    2>&1 | tee "$RESULT_DIR/${label}-collective-health.log"
}

monitor_host() {
  local output="$1"
  while :; do
    local running_name
    memory_snapshot sample
    cat /proc/pressure/memory
    running_name="$(docker ps --filter "name=${CONTAINER_PREFIX}-" --format '{{.Names}}' | head -1)"
    if [ -n "$running_name" ]; then
      docker stats --no-stream --format \
        'docker name={{.Name}} mem={{.MemUsage}} mem_percent={{.MemPerc}} pids={{.PIDs}}' \
        "$running_name" 2>/dev/null || true
    fi
    sleep 5
  done >"$output" 2>&1
}

wait_ready() {
  local log="$1" start now sig last_sig last_progress
  start="$(date +%s)"
  last_progress="$start"
  last_sig=""
  while :; do
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    kill -0 "$server_pid" 2>/dev/null || {
      echo "server exited before readiness" >&2
      tail -120 "$log" >&2 || true
      return 1
    }
    now="$(date +%s)"
    sig="$(sha256sum "$log" | awk '{print $1}')"
    if [ "$sig" != "$last_sig" ]; then
      last_sig="$sig"
      last_progress="$now"
    fi
    [ "$((now - last_progress))" -lt "$READY_STALL" ] || {
      echo "server readiness stalled for ${READY_STALL}s" >&2
      tail -120 "$log" >&2 || true
      return 1
    }
    [ "$((now - start))" -lt "$READY_TIMEOUT" ] || {
      echo "server readiness exceeded ${READY_TIMEOUT}s" >&2
      tail -120 "$log" >&2 || true
      return 1
    }
    sleep 5
  done
}

echo "CONFIG"
env STAMP="$STAMP" RESULT_DIR="$RESULT_DIR" CACHE_ROOT="$CACHE_ROOT" \
  "$0" --print-config | tee "$RESULT_DIR/config.txt"
host_gate
memory_snapshot pre | tee "$RESULT_DIR/memory-pre.txt"
"$LAUNCHER" --verify-image | tee "$RESULT_DIR/image-verify.txt"

python3 "$MODEL_VERIFY" "$MODEL_MANIFEST" "$MODEL_DIR" \
  --json "$RESULT_DIR/model-verify.json" >"$RESULT_DIR/model-verify.log"
echo "model direct-and-ordinary identity -> pass files=66 bytes=30866866928"

uname -a >"$RESULT_DIR/uname.txt"
dpkg-query -W intel-opencl-icd libze1 2>/dev/null >"$RESULT_DIR/host-packages.txt" || true
docker image inspect neural-download/vllm-openai-xpu:qwen38-fp8-collective-work-wait-r15 \
  >"$RESULT_DIR/image-inspect.json"

echo "COMMAND"
echo "bin/gpu-run bash vllm/fp8/qualify_qwen38_fp8_neural_f02.sh --leased"
echo "pre-health -> begin"
run_health pre
echo "pre-health -> pass"

monitor_host "$RESULT_DIR/host-monitor.log" &
monitor_pid=$!

for attempt in $(seq 1 "$ATTEMPTS"); do
  host_gate
  attempt_dir="$RESULT_DIR/attempt-$attempt"
  if [ "$SHARED_CACHE" -eq 1 ]; then
    cache_dir="$CACHE_ROOT/shared"
  else
    cache_dir="$CACHE_ROOT/attempt-$attempt"
  fi
  current_name="${CONTAINER_PREFIX}-${STAMP}-${attempt}"
  mkdir -p "$attempt_dir"
  echo "attempt=$attempt server -> start"
  env MODEL_DIR="$MODEL_DIR" CACHE_DIR="$cache_dir" NAME="$current_name" \
    ALLOW_EXISTING_CACHE="$SHARED_CACHE" \
    SERVED="$SERVED" PORT="$PORT" "$LAUNCHER" run \
    >"$attempt_dir/server.log" 2>&1 &
  server_pid=$!
  wait_ready "$attempt_dir/server.log"

  curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/v1/models" \
    >"$attempt_dir/models.json"
  python3 - "$attempt_dir/models.json" "$SERVED" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
ids = [item["id"] for item in data["data"]]
assert ids == [sys.argv[2]], ids
PY
  docker inspect "$current_name" >"$attempt_dir/container-inspect.json"
  docker exec "$current_name" sha256sum \
    /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py \
    /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py \
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
    >"$attempt_dir/runtime-files.sha256"
  docker stats --no-stream --format '{{json .}}' "$current_name" \
    >"$attempt_dir/docker-stats-before.json"

  python3 "$BENCH" \
    --base-url "http://127.0.0.1:${PORT}" --model "$SERVED" \
    --api-mode completions --suite "$SUITE" \
    --max-tokens 512 --metric-tokens 100 --seed 42 --timeout 900 \
    --return-token-ids --require-natural-eos \
    --request-extra-json '{"temperature":0,"top_p":1}' \
    --out "$attempt_dir/performance.json" \
    >"$attempt_dir/performance.stdout"
  python3 "$CANARIES" \
    --base-url "http://127.0.0.1:${PORT}" --model "$SERVED" \
    --out "$attempt_dir/canaries.json" >"$attempt_dir/canaries.stdout"

  curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/health" \
    >"$attempt_dir/endpoint-post-health.json"
  docker stats --no-stream --format '{{json .}}' "$current_name" \
    >"$attempt_dir/docker-stats-after.json"
  docker logs "$current_name" >"$attempt_dir/server-final.log" 2>&1 || true
  stop_server
  docker inspect "$current_name" >/dev/null 2>&1 && {
    echo "attempt=$attempt teardown failed" >&2
    exit 1
  }
  ss -ltn | grep -Eq ":${PORT}[[:space:]]" && {
    echo "attempt=$attempt port remained occupied" >&2
    exit 1
  }
  echo "attempt=$attempt post-health -> begin"
  run_health "attempt-${attempt}-post"
  echo "attempt=$attempt post-health -> pass"
done

stop_monitor
memory_snapshot post | tee "$RESULT_DIR/memory-post.txt"
journalctl -k --since "@${journal_start}" --no-pager >"$RESULT_DIR/kernel-journal.log"

set +e
reference_gate_args=()
if [ "$REQUIRE_REFERENCE_EXACT" -eq 1 ]; then
  reference_gate_args=(--require-reference-exact)
fi
python3 "$SCRIPT_DIR/analyze_qwen38_fp8_f02.py" \
  --result-dir "$RESULT_DIR" --attempts "$ATTEMPTS" \
  --served-model "$SERVED" \
  --publisher-attempt "$PUBLISHER_A" --publisher-attempt "$PUBLISHER_B" \
  --schema "$ANALYZER_SCHEMA" --completion-route "$COMPLETION_ROUTE" \
  "${reference_gate_args[@]}" \
  --output "$RESULT_DIR/summary.json"
analysis_rc=$?
set -e

if grep -Eqi 'xe 0000:(43|47):00\.0.*(reset|fault|timeout|timed out|fatal|wedged|failed)' \
  "$RESULT_DIR/kernel-journal.log"; then
  echo "new GPU kernel fault event detected" >&2
  exit 1
fi
max_swap_kib="$(awk -F'[ =]' '/swap_used_kib=/ {for(i=1;i<=NF;i++) if($i=="swap_used_kib") print $(i+1)}' "$RESULT_DIR/host-monitor.log" | sort -n | tail -1)"
max_swap_kib="${max_swap_kib:-0}"
[ "$max_swap_kib" -le "$((MAX_SWAP_USED_MIB * 1024))" ] || {
  echo "host swap gate failed: max_swap_used_kib=$max_swap_kib" >&2
  exit 1
}

echo "RESULT"
cat "$RESULT_DIR/summary.json"
echo "max_host_swap_used_kib=$max_swap_kib"
echo "VERDICT"
if [ "$analysis_rc" -eq 0 ]; then
  echo "$CAMPAIGN_LABEL passed cross-server exactness under the local P2P-off no-swap safety port."
else
  echo "$CAMPAIGN_LABEL failed cross-server raw-token exactness."
fi
echo "result_dir=$RESULT_DIR"
echo "$analysis_rc" >"$RESULT_DIR/qualifier.rc"
trap - EXIT INT TERM HUP
exit "$analysis_rc"
