#!/usr/bin/env bash
# Reproduce Steve Seguin's R187 graph-off profile or documented slow-host
# XPU-graph variant with local lease, identity, health, and teardown gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260904-source-r187}"
SOURCE_COMMIT="${SOURCE_COMMIT:-8319e0964df12a1f0bc920301efc662ac49a949e}"
PACKAGE="$SOURCE/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70"
IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-gdn-split-mixed-r156}"
IMAGE_ID="${IMAGE_ID:-sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152}"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
PROFILE="${PROFILE:-r187}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/qwen38_fp8_steve_${PROFILE}/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/qwen38_fp8_steve_${PROFILE}/$STAMP}"
PORT="${PORT:-18124}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
READY_STALL="${READY_STALL:-360}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-64}"

case "$PROFILE" in
  r156)
    LAUNCHER="$SOURCE/experiments/qwen38-27b-b70/scripts/run-20260903-qwen38-fp8-mtp1-split-mixed-r156-server.sh"
    SERVED="qwen38-fp8-block-w8a16-mtp1-split-mixed-r156"
    PROFILE_LABEL="mtp1-split-mixed-r156-local"
    ;;
  r187)
    LAUNCHER="$SOURCE/experiments/qwen38-27b-b70/scripts/run-20260903-qwen38-fp8-mtp1-whole-graph-r187-server.sh"
    SERVED="qwen38-fp8-block-w8a16-mtp1-whole-graph-r187"
    PROFILE_LABEL="mtp1-whole-graph-r187-local"
    ;;
  xpugraph-r156)
    LAUNCHER="$PACKAGE/run-w8a16-mtp1-strict-server-xpugraph.sh"
    SERVED="qwen38-fp8-block-w8a16-mtp1-xpugraph-r156-local"
    PROFILE_LABEL="mtp1-xpugraph-r156-local"
    ;;
  *)
    printf 'PROFILE must be r156, r187, or xpugraph-r156\n' >&2
    exit 2
    ;;
esac

NAME="${NAME:-qwen38-fp8-steve-${PROFILE}-${STAMP}}"
BENCH="$PACKAGE/bench-w8a16-mtp1-strict.sh"

case "${1:-}" in
  --leased) shift ;;
  --print-config)
    printf '%s\n' \
      "source=$SOURCE" \
      "source_commit=$SOURCE_COMMIT" \
      "image=$IMAGE" \
      "image_id=$IMAGE_ID" \
      "model_dir=$MODEL_DIR" \
      "profile=$PROFILE" \
      "result_dir=$RESULT_DIR" \
      "cache_dir=$CACHE_DIR" \
      "container=$NAME" \
      "port=$PORT" \
      "served_model=$SERVED"
    exit 0
    ;;
  '')
    exec env B70_AGENT="qwen38-fp8-steve-${PROFILE}" \
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

[[ ! -e "$RESULT_DIR" ]] || { printf 'RESULT_DIR must be new: %s\n' "$RESULT_DIR" >&2; exit 1; }
[[ ! -e "$CACHE_DIR" ]] || { printf 'CACHE_DIR must be new: %s\n' "$CACHE_DIR" >&2; exit 1; }
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  printf 'source checkout is not at %s\n' "$SOURCE_COMMIT" >&2
  exit 1
}
for required in "$MODEL_DIR" "$LAUNCHER" "$BENCH"; do
  [[ -e "$required" ]] || { printf 'missing input: %s\n' "$required" >&2; exit 1; }
done
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] || {
  printf 'image ID mismatch\n' >&2
  exit 1
}

mkdir -p "$RESULT_DIR"
journal_start="$(date +%s)"
server_pid=""
monitor_pid=""

stop_server() {
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [[ -n "$server_pid" ]]; then wait "$server_pid" 2>/dev/null || true; fi
  server_pid=""
}

stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  monitor_pid=""
}

run_health() {
  local label=$1
  "$REPO/bin/xpu-health" --img "$IMAGE" 2>&1 \
    | tee "$RESULT_DIR/${label}-card-health.log"
  env IMG="$IMAGE" "$REPO/bin/xpu-collective-health" \
    --p2p 0 --timeout "$HEALTH_TIMEOUT" 2>&1 \
    | tee "$RESULT_DIR/${label}-collective-health.log"
}

cleanup() {
  local rc=$?
  set +e
  stop_server
  stop_monitor
  journalctl -k --since "@$journal_start" --no-pager \
    >"$RESULT_DIR/kernel-journal.log" 2>"$RESULT_DIR/kernel-journal.err"
  if [[ "$rc" -ne 0 ]]; then
    "$REPO/bin/xe-reset" --method rebind >"$RESULT_DIR/recovery.log" 2>&1
    run_health failure-post >"$RESULT_DIR/failure-post.log" 2>&1
  fi
  printf '%s\n' "$rc" >"$RESULT_DIR/qualifier.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "$available_kib" -ge "$((MIN_AVAILABLE_GIB * 1024 * 1024))" ]] || {
  printf 'host gate failed: MemAvailable below %s GiB\n' "$MIN_AVAILABLE_GIB" >&2
  exit 1
}
swap_used_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
[[ "$swap_used_kib" -le 1048576 ]] || {
  printf 'host gate failed: swap use above 1 GiB\n' >&2
  exit 1
}

monitor_host() {
  while :; do
    awk '/^MemAvailable:/ {print "mem_available_kib=" $2}' /proc/meminfo
    awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print "swap_used_kib=" t-f}' /proc/meminfo
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
    if [[ "$signature" != "$last_signature" ]]; then
      last_signature="$signature"
      last_progress="$now"
    fi
    [[ "$((now - last_progress))" -lt "$READY_STALL" ]] || return 1
    [[ "$((now - start))" -lt "$READY_TIMEOUT" ]] || return 1
    sleep 5
  done
}

echo 'CONFIG'
env "$0" --print-config | tee "$RESULT_DIR/config.txt"
echo "COMMAND -> bin/gpu-run bash $0 --leased" | tee "$RESULT_DIR/command.txt"
env \
  EXPECTED_XPU_EXTENSION_SHA256=f912e12de1d79206221142c9a50af2aba70d2c77c735c9cd2d5d8d9def0740d1 \
  EXPECTED_XPU_OPS_SHA256=6a7761930cd8b9e3f67902648ba5aaaf708567cebf70fcedda595d698f26b064 \
  "$PACKAGE/verify-image-contract.sh" mtp1-serial-fa-split-gdn "$IMAGE" \
  | tee "$RESULT_DIR/image-contract.txt"
docker image inspect "$IMAGE" >"$RESULT_DIR/image-inspect.json"
run_health pre

monitor_host &
monitor_pid=$!
env \
  IMAGE="$IMAGE" EXPECTED_IMAGE_ID="$IMAGE_ID" \
  MODEL_DIR="$MODEL_DIR" VLLM_CACHE_DIR="$CACHE_DIR" \
  CONTAINER_NAME="$NAME" PORT="$PORT" SERVED_MODEL_NAME="$SERVED" \
  MAX_MODEL_LEN=1024 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=1024 \
  CONTAINER_MEMORY=12g CONTAINER_MEMORY_SWAP=16g \
  EXPECTED_XPU_EXTENSION_SHA256=f912e12de1d79206221142c9a50af2aba70d2c77c735c9cd2d5d8d9def0740d1 \
  EXPECTED_XPU_OPS_SHA256=6a7761930cd8b9e3f67902648ba5aaaf708567cebf70fcedda595d698f26b064 \
  VLLM_XPU_GDN_SPLIT_MIXED=1 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4=1 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE=128 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_SCALE_DTYPE=bf16 \
  VLLM_XPU_DRAFT_LM_HEAD_INT4_CHUNK_ROWS=2048 \
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
  MODEL_NAME="$SERVED" PROFILE_LABEL="$PROFILE_LABEL" \
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

python3 - "$RESULT_DIR/attempt-1/performance.json" "$RESULT_DIR/verdict.json" "$PROFILE" <<'PY'
import json
import pathlib
import sys

performance = json.load(open(sys.argv[1], encoding="ascii"))
rate = performance["summary"]["class_balanced_tok_s_1_100_intervals_after_ttft"]["median"]
verdict = {
    "schema": "b70.qwen38-fp8-steve-r187-strict.v1",
    "profile": sys.argv[3],
    "local_tok_s": rate,
    "strict_workload_gate": performance["realistic_final_gate"]["passed"],
    "cached_tokens_all_zero": performance["fresh_response_validity"]["cached_tokens_all_zero"],
    "single_attempt_only": True,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="ascii"
)
print(json.dumps(verdict, indent=2, sort_keys=True))
assert verdict["strict_workload_gate"]
assert verdict["cached_tokens_all_zero"]
PY

if grep -Eqi 'xe 0000:(0b|44):00\.0.*(reset|fault|timeout|timed out|fatal|wedged|failed)' \
  "$RESULT_DIR/kernel-journal.log"; then
  printf 'new GPU kernel fault event detected\n' >&2
  exit 1
fi

printf 'RESULT\n'
cat "$RESULT_DIR/verdict.json"
printf 'VERDICT -> strict workload, identity, health, and teardown passed; cross-attempt parity remains open\n'
printf '0\n' >"$RESULT_DIR/qualifier.rc"
trap - EXIT INT TERM HUP
