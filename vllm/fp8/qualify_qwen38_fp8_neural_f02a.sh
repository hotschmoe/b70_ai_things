#!/usr/bin/env bash
# Repeat the five F02-sensitive prompts twice inside one fresh server lifetime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source}"
SOURCE_COMMIT="0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
SUITE="$SOURCE/repro/qwen36-27b-autoround-int4-b70/realistic-suite-v1.json"
SUITE_SHA256="df03f49d36c36d2b8ac4cd117b7cb2e42c74878af1f6926690ebb89eeccd47ac"
BENCH="$SOURCE/scripts/bench-openai-realistic-suite.py"
LAUNCHER="$SCRIPT_DIR/serve_qwen38_fp8_neural_f02.sh"
F02_ROOT="${F02_ROOT:-$ROOT/results/f02_qwen38_fp8_neural/20260829T231100Z}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/f02a_qwen38_fp8_neural/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f02a_qwen38_fp8_neural/$STAMP}"
SERVED="qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02a"
NAME="qwen38-fp8-neural-f02a-${STAMP}"
PORT="${PORT:-18188}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
READY_STALL="${READY_STALL:-360}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-96}"

PROMPTS=(
  incident-retrospective
  code-review
  customer-email
  performance-hypotheses
  decision-memo
)

case "${1:-}" in
  --leased) shift ;;
  --print-config)
    echo "source=$SOURCE"
    echo "source_commit=$SOURCE_COMMIT"
    echo "model_dir=$MODEL_DIR"
    echo "suite=$SUITE"
    echo "f02_reference=$F02_ROOT"
    echo "result_dir=$RESULT_DIR"
    echo "cache_dir=$CACHE_DIR"
    echo "served_model=$SERVED"
    echo "container=$NAME"
    echo "port=$PORT"
    echo "p2p=0"
    echo "mtp=0"
    echo "xpu_graph=0"
    echo "repeats=2"
    echo "prompt_ids=${PROMPTS[*]}"
    exit 0
    ;;
  '')
    exec env B70_AGENT=f02a-qwen38-fp8-neural \
      "$REPO/bin/gpu-run" bash "$0" --leased
    ;;
  *) echo "usage: $0 [--print-config]" >&2; exit 2 ;;
esac

for pair in "PORT:$PORT" "READY_TIMEOUT:$READY_TIMEOUT" \
  "READY_STALL:$READY_STALL" "HEALTH_TIMEOUT:$HEALTH_TIMEOUT" \
  "MIN_AVAILABLE_GIB:$MIN_AVAILABLE_GIB"; do
  value="${pair#*:}"
  case "$value" in ''|*[!0-9]*|0) echo "${pair%%:*} must be positive" >&2; exit 2 ;; esac
done
[ ! -e "$RESULT_DIR" ] || { echo "RESULT_DIR must be new: $RESULT_DIR" >&2; exit 1; }
[ ! -e "$CACHE_DIR" ] || { echo "CACHE_DIR must be new: $CACHE_DIR" >&2; exit 1; }
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  echo "source checkout is not at $SOURCE_COMMIT" >&2
  exit 1
}
for required in "$MODEL_DIR" "$SUITE" "$BENCH" "$LAUNCHER" \
  "$F02_ROOT/summary.json" "$F02_ROOT/attempt-1/performance.json" \
  "$F02_ROOT/attempt-2/performance.json"; do
  [ -e "$required" ] || { echo "missing input: $required" >&2; exit 1; }
done
[ "$(sha256sum "$SUITE" | awk '{print $1}')" = "$SUITE_SHA256" ] || {
  echo "suite hash mismatch" >&2
  exit 1
}
jq -e '.verdict == "failed_cross_server_token_exactness"' \
  "$F02_ROOT/summary.json" >/dev/null

mkdir -p "$RESULT_DIR"
journal_start="$(date +%s)"
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

host_gate() {
  local available
  available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  [ "$available" -ge "$((MIN_AVAILABLE_GIB * 1024 * 1024))" ] || {
    echo "host gate failed: MemAvailable below ${MIN_AVAILABLE_GIB} GiB" >&2
    return 1
  }
  [ "$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)" -le 1048576 ] || {
    echo "host gate failed: swap use above 1 GiB" >&2
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
  local available swap_total swap_free
  while :; do
    available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    swap_total="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    swap_free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    echo "mem_available_kib=$available swap_used_kib=$((swap_total - swap_free))"
    cat /proc/pressure/memory
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
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    kill -0 "$server_pid" 2>/dev/null || {
      echo "server exited before readiness" >&2
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
      echo "server readiness stalled" >&2
      return 1
    }
    [ "$((now - start))" -lt "$READY_TIMEOUT" ] || {
      echo "server readiness timeout" >&2
      return 1
    }
    sleep 5
  done
}

echo "CONFIG"
env STAMP="$STAMP" RESULT_DIR="$RESULT_DIR" CACHE_DIR="$CACHE_DIR" \
  F02_ROOT="$F02_ROOT" "$0" --print-config | tee "$RESULT_DIR/config.txt"
host_gate
"$LAUNCHER" --verify-image | tee "$RESULT_DIR/image-verify.txt"
echo "COMMAND"
echo "bin/gpu-run bash vllm/fp8/qualify_qwen38_fp8_neural_f02a.sh --leased"
run_health pre

monitor_host &
monitor_pid=$!
env MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" NAME="$NAME" \
  SERVED="$SERVED" PORT="$PORT" "$LAUNCHER" run \
  >"$RESULT_DIR/server.log" 2>&1 &
server_pid=$!
wait_ready

curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/v1/models" >"$RESULT_DIR/models.json"
python3 - "$RESULT_DIR/models.json" "$SERVED" <<'PY'
import json
import sys
ids = [item["id"] for item in json.load(open(sys.argv[1], encoding="ascii"))["data"]]
assert ids == [sys.argv[2]], ids
PY
docker inspect "$NAME" >"$RESULT_DIR/container-inspect.json"
grep -Fq 'Selected XPUFp8BlockScaledMMKernel for Fp8LinearMethod' "$RESULT_DIR/server.log"
grep -Fq "'cudagraph_mode': <CUDAGraphMode.NONE: 0>" "$RESULT_DIR/server.log"
grep -Fq 'CCL_TOPO_P2P_ACCESS changed to be 0' "$RESULT_DIR/server.log"
grep -Fq 'dtype=torch.float16' "$RESULT_DIR/server.log"

prompt_args=()
for prompt in "${PROMPTS[@]}"; do prompt_args+=(--prompt-id "$prompt"); done
for repeat in 1 2; do
  python3 "$BENCH" \
    --base-url "http://127.0.0.1:${PORT}" --model "$SERVED" \
    --api-mode completions --suite "$SUITE" \
    --max-tokens 512 --metric-tokens 100 --seed 42 --timeout 900 \
    --return-token-ids --require-natural-eos --allow-screening \
    --request-extra-json '{"temperature":0,"top_p":1}' \
    "${prompt_args[@]}" --out "$RESULT_DIR/repeat-$repeat.json" \
    >"$RESULT_DIR/repeat-$repeat.stdout"
done

curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/health" \
  >"$RESULT_DIR/endpoint-post-health.json"
docker stats --no-stream --format '{{json .}}' "$NAME" >"$RESULT_DIR/docker-stats.json"
docker logs "$NAME" >"$RESULT_DIR/server-final.log" 2>&1 || true
stop_server
stop_monitor
run_health post
journalctl -k --since "@${journal_start}" --no-pager >"$RESULT_DIR/kernel-journal.log"

python3 - "$RESULT_DIR" "$F02_ROOT" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
f02 = Path(sys.argv[2])
left = json.loads((root / "repeat-1.json").read_text(encoding="ascii"))
right = json.loads((root / "repeat-2.json").read_text(encoding="ascii"))
references = [
    json.loads((f02 / f"attempt-{index}" / "performance.json").read_text(encoding="ascii"))
    for index in (1, 2)
]

def digest(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()

def compare(a, b):
    first = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {
        "exact": a == b,
        "first_mismatch_zero_based": first,
        "mismatch_count": sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)),
        "left_count": len(a),
        "right_count": len(b),
        "left_sha256": digest(a),
        "right_sha256": digest(b),
    }

left_by_id = {row["prompt_id"]: row for row in left["rows"]}
right_by_id = {row["prompt_id"]: row for row in right["rows"]}
prompt_ids = list(left_by_id)
assert prompt_ids == list(right_by_id)
assert len(prompt_ids) == 5
assert all(row["cached_tokens"] == 0 for row in left["rows"] + right["rows"])
within = []
for prompt_id in prompt_ids:
    item = compare(left_by_id[prompt_id]["token_ids"], right_by_id[prompt_id]["token_ids"])
    item["prompt_id"] = prompt_id
    within.append(item)

reference_results = []
for reference_index, reference in enumerate(references, start=1):
    reference_by_id = {row["prompt_id"]: row for row in reference["rows"]}
    for repeat_index, repeat in enumerate((left_by_id, right_by_id), start=1):
        comparisons = []
        for prompt_id in prompt_ids:
            item = compare(repeat[prompt_id]["token_ids"], reference_by_id[prompt_id]["token_ids"])
            item["prompt_id"] = prompt_id
            comparisons.append(item)
        reference_results.append({
            "repeat": repeat_index,
            "f02_attempt": reference_index,
            "exact_prompts": sum(item["exact"] for item in comparisons),
            "comparisons": comparisons,
        })

rates = [
    document["summary"]["class_balanced_tok_s_1_100_intervals_after_ttft"]["median"]
    for document in (left, right)
]
exact = sum(item["exact"] for item in within)
summary = {
    "schema": "b70.qwen38-fp8-neural-f02a.v1",
    "verdict": "within_lifetime_exact" if exact == 5 else "within_lifetime_nondeterministic",
    "within_lifetime_exact_prompts": exact,
    "total_prompts": 5,
    "within_lifetime_comparisons": within,
    "f02_reference_comparisons": reference_results,
    "cached_tokens_all_zero": True,
    "diagnostic_rates": rates,
    "diagnostic_rate_median": statistics.median(rates),
    "performance_attribution_qualified": False,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if grep -Eqi 'xe 0000:(43|47):00\.0.*(reset|fault|timeout|timed out|fatal|wedged|failed)' \
  "$RESULT_DIR/kernel-journal.log"; then
  echo "new GPU kernel fault event detected" >&2
  exit 1
fi
max_swap_kib="$(awk -F= '/swap_used_kib=/ {print $3}' "$RESULT_DIR/host-monitor.log" | sort -n | tail -1)"
max_swap_kib="${max_swap_kib:-0}"
[ "$max_swap_kib" -eq 0 ] || {
  echo "F02a used host swap: max_swap_used_kib=$max_swap_kib" >&2
  exit 1
}
echo "RESULT"
cat "$RESULT_DIR/summary.json"
echo "VERDICT"
jq -r '.verdict' "$RESULT_DIR/summary.json"
echo "result_dir=$RESULT_DIR"
echo 0 >"$RESULT_DIR/qualifier.rc"
trap - EXIT INT TERM HUP
