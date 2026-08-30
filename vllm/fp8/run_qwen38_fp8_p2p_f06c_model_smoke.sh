#!/usr/bin/env bash
# F06c/F06d: one-server, one-request direct-P2P model-loaded smoke transaction.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source}"
IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81}"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f05d_qwen38_fp8_neural/20260830T075200Z/attempt-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/f06c_qwen38_fp8_neural_p2p/$STAMP}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f06c_qwen38_fp8_neural_p2p/$STAMP}"
NAME="${NAME:-qwen38-fp8-neural-f06c-p2p1-$STAMP}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-0}"
CAMPAIGN_LABEL="${CAMPAIGN_LABEL:-F06c}"
SERVED="${SERVED:-qwen3.8-27b-FP8-official-W8A16-mtp${SPECULATIVE_TOKENS}-p2p1-fp16kv-f06c}"
PORT="${PORT:-18191}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
READY_STALL="${READY_STALL:-600}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
EXTRA_SMOKE="${EXTRA_SMOKE:-}"
LAUNCHER="$SCRIPT_DIR/serve_qwen38_fp8_neural_f02.sh"
MODEL_MANIFEST="$SOURCE/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/model-direct.json"
MODEL_VERIFY="$SOURCE/repro/qwen38-27b-autoround-int4-b70/scripts/verify-model-direct.py"

if [ "${1:-}" != --leased ]; then
  exec "$REPO/bin/gpu-run" env \
    I_KNOW_P2P_WEDGES="${I_KNOW_P2P_WEDGES:-0}" \
    STAMP="$STAMP" RESULT_DIR="$RESULT_DIR" CACHE_DIR="$CACHE_DIR" \
    NAME="$NAME" SERVED="$SERVED" PORT="$PORT" \
    bash "$0" --leased
fi

if [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing F06c direct P2P without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi
case "$SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) echo "SPECULATIVE_TOKENS must be a nonnegative integer" >&2; exit 2 ;;
esac
for required in "$MODEL_DIR" "$SEED_CACHE" "$LAUNCHER" "$MODEL_MANIFEST" "$MODEL_VERIFY"; do
  [ -e "$required" ] || { echo "missing input: $required" >&2; exit 1; }
done
[ -z "$EXTRA_SMOKE" ] || [ -f "$EXTRA_SMOKE" ] || {
  echo "EXTRA_SMOKE is not a file: $EXTRA_SMOKE" >&2
  exit 1
}
[ ! -e "$RESULT_DIR" ] || { echo "RESULT_DIR must be new: $RESULT_DIR" >&2; exit 1; }
[ ! -e "$CACHE_DIR" ] || { echo "CACHE_DIR must be new: $CACHE_DIR" >&2; exit 1; }
docker inspect "$NAME" >/dev/null 2>&1 && { echo "container already exists: $NAME" >&2; exit 1; }

mkdir -p "$RESULT_DIR" "$CACHE_DIR"
server_pid=""
cleanup() {
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [ -n "$server_pid" ]; then wait "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

memory_snapshot() {
  local label="$1" available total free
  available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  total="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  free="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  echo "$label mem_available_kib=$available swap_used_kib=$((total - free))"
}

mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
swap_total_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
[ "$mem_available_kib" -ge "$((96 * 1024 * 1024))" ] || {
  echo "host gate failed: MemAvailable below 96 GiB" >&2
  exit 1
}
[ "$((swap_total_kib - swap_free_kib))" -le "$((1024 * 1024))" ] || {
  echo "host gate failed: swap use above 1024 MiB" >&2
  exit 1
}

run_logged() {
  local label="$1"
  shift
  echo "COMMAND -> $label: $*" | tee -a "$RESULT_DIR/commands.txt"
  "$@" 2>&1 | tee "$RESULT_DIR/$label.log"
  return "${PIPESTATUS[0]}"
}

wait_ready() {
  local started now last_size size last_progress
  started="$(date +%s)"
  last_progress="$started"
  last_size=0
  while :; do
    curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    kill -0 "$server_pid" 2>/dev/null || return 1
    now="$(date +%s)"
    size="$(stat -c %s "$RESULT_DIR/server.log" 2>/dev/null || echo 0)"
    if [ "$size" != "$last_size" ]; then
      last_size="$size"
      last_progress="$now"
    fi
    [ "$((now - started))" -lt "$READY_TIMEOUT" ] || return 1
    [ "$((now - last_progress))" -lt "$READY_STALL" ] || return 1
    sleep 5
  done
}

actual_image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
[ "$actual_image_id" = "$EXPECTED_IMAGE_ID" ] || {
  echo "image ID mismatch: actual=$actual_image_id expected=$EXPECTED_IMAGE_ID" >&2
  exit 1
}
python3 "$MODEL_VERIFY" "$MODEL_MANIFEST" "$MODEL_DIR" \
  --json "$RESULT_DIR/model-verify.json" >"$RESULT_DIR/model-verify.log"
docker run --rm --volume "$SEED_CACHE:/seed:ro" --volume "$CACHE_DIR:/dest" \
  --entrypoint cp "$IMAGE" -a /seed/. /dest/

{
  echo "CONFIG -> image=$IMAGE"
  echo "CONFIG -> image_id=$actual_image_id"
  echo "CONFIG -> kernel=$(uname -r)"
  echo "CONFIG -> model=$MODEL_DIR"
  echo "CONFIG -> served=$SERVED"
  echo "CONFIG -> tp=2 p2p=1 mtp=$SPECULATIVE_TOKENS xpu_graph=0 compilation=PIECEWISE"
  echo "CONFIG -> max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "CONFIG -> extra_smoke=${EXTRA_SMOKE:-none}"
  echo "CONFIG -> dtype=float16 quantization=fp8 kv_cache_dtype=auto"
  echo "CONFIG -> container_memory_gib=32 container_swap_extra_gib=0"
  echo "CONFIG -> seed_cache=$SEED_CACHE"
  echo "CONFIG -> result_dir=$RESULT_DIR"
} | tee "$RESULT_DIR/config.txt"
memory_snapshot pre | tee "$RESULT_DIR/memory-pre.txt"

pre_card_rc=0
pre_p2p0_rc=0
transaction_rc=0
recovery_rc=0
post_card_rc=0
post_p2p0_rc=0
run_logged pre-card "$REPO/bin/xpu-health" --img "$IMAGE" || pre_card_rc=$?
run_logged pre-p2p0 env IMG="$IMAGE" \
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 || pre_p2p0_rc=$?
if [ "$pre_card_rc" -ne 0 ] || [ "$pre_p2p0_rc" -ne 0 ]; then
  echo "VERDICT -> blocked: pre-health failed card=$pre_card_rc p2p0=$pre_p2p0_rc" \
    | tee "$RESULT_DIR/verdict.txt"
  exit 1
fi

journal_start="$(date +%s)"
echo "COMMAND -> start exact MTP${SPECULATIVE_TOKENS} model with guarded direct P2P" | tee -a "$RESULT_DIR/commands.txt"
env I_KNOW_P2P_WEDGES=1 P2P_ACCESS=1 \
  IMAGE="$IMAGE" EXPECTED_IMAGE_ID="$EXPECTED_IMAGE_ID" \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" ALLOW_EXISTING_CACHE=1 \
  NAME="$NAME" SERVED="$SERVED" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  SPECULATIVE_TOKENS="$SPECULATIVE_TOKENS" RMS_PACKED_SERIAL_EXACT=1 GDN_PERSISTENT_SCRATCH=1 \
  INDUCTOR_COMBO_KERNELS=0 INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 INDUCTOR_DETERMINISTIC_CONFIG=1 \
  "$LAUNCHER" run >"$RESULT_DIR/server.log" 2>&1 &
server_pid=$!

if ! wait_ready; then
  transaction_rc=1
  echo "RESULT -> server failed or stalled before readiness" | tee -a "$RESULT_DIR/verdict.txt"
  tail -160 "$RESULT_DIR/server.log" | tee "$RESULT_DIR/server-tail.log"
else
  echo "RESULT -> server ready" | tee -a "$RESULT_DIR/verdict.txt"
  curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/v1/models" \
    >"$RESULT_DIR/models.json" || transaction_rc=$?
  if [ "$transaction_rc" -eq 0 ]; then
    python3 - "$RESULT_DIR/models.json" "$SERVED" <<'PY' || transaction_rc=$?
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
assert [item["id"] for item in data["data"]] == [sys.argv[2]], data
PY
  fi
  if [ "$transaction_rc" -eq 0 ] && [ -n "$EXTRA_SMOKE" ]; then
    python3 "$EXTRA_SMOKE" \
      --base-url "http://127.0.0.1:${PORT}" --model "$SERVED" \
      --concurrency 4 --rounds 8 --timeout 600 --seed 42 \
      --output-json "$RESULT_DIR/concurrent-quality.json" \
      >"$RESULT_DIR/concurrent-quality.stdout" 2>&1 || transaction_rc=$?
    if [ "$transaction_rc" -eq 0 ]; then
      python3 - "$RESULT_DIR/concurrent-quality.json" <<'PY' || transaction_rc=$?
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
assert data["pass_all"] is True, data
assert data["total_requests"] == 32, data
PY
    fi
  fi
  if [ "$transaction_rc" -eq 0 ]; then
    python3 - "$PORT" "$SERVED" "$RESULT_DIR/smoke.json" <<'PY' || transaction_rc=$?
import json
import sys
import urllib.request

port, served, output = sys.argv[1:]
payload = json.dumps({
    "model": served,
    "prompt": "Write exactly the word READY and nothing else.",
    "max_tokens": 16,
    "temperature": 0,
    "top_p": 1,
}).encode("ascii")
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    data = json.load(response)
assert data["model"] == served, data
assert len(data["choices"]) == 1 and data["choices"][0]["text"], data
assert data["usage"]["completion_tokens"] > 0, data
with open(output, "w", encoding="ascii") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=True, sort_keys=True)
    handle.write("\n")
PY
  fi
  curl -fsS --max-time 15 "http://127.0.0.1:${PORT}/health" \
    >"$RESULT_DIR/endpoint-post-health.json" || transaction_rc=$?
  docker inspect "$NAME" >"$RESULT_DIR/container-inspect.json" 2>/dev/null || transaction_rc=$?
  docker stats --no-stream --format '{{json .}}' "$NAME" \
    >"$RESULT_DIR/docker-stats.json" 2>/dev/null || transaction_rc=$?
fi

docker logs "$NAME" >"$RESULT_DIR/server-final.log" 2>&1 || true
docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
wait "$server_pid" 2>/dev/null || true
server_pid=""
docker rm -f "$NAME" >/dev/null 2>&1 || true
if [ "$transaction_rc" -ne 0 ]; then
  run_logged recovery-rebind "$REPO/bin/xe-reset" --method rebind || recovery_rc=$?
fi

run_logged post-card "$REPO/bin/xpu-health" --img "$IMAGE" || post_card_rc=$?
run_logged post-p2p0 env IMG="$IMAGE" \
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 || post_p2p0_rc=$?
journalctl -k --since "@${journal_start}" --no-pager >"$RESULT_DIR/kernel-journal.log"
memory_snapshot post | tee "$RESULT_DIR/memory-post.txt"

p2p_log_rc=0
grep -Fq 'CCL_TOPO_P2P_ACCESS changed to be 1' "$RESULT_DIR/server.log" || p2p_log_rc=$?
{
  echo "RESULT -> pre_card_rc=$pre_card_rc pre_p2p0_rc=$pre_p2p0_rc"
  echo "RESULT -> transaction_rc=$transaction_rc p2p_log_rc=$p2p_log_rc recovery_rc=$recovery_rc"
  echo "RESULT -> post_card_rc=$post_card_rc post_p2p0_rc=$post_p2p0_rc"
  if [ "$transaction_rc" -eq 0 ] && [ "$p2p_log_rc" -eq 0 ] && \
     [ "$post_card_rc" -eq 0 ] && [ "$post_p2p0_rc" -eq 0 ]; then
    echo "VERDICT -> $CAMPAIGN_LABEL PASS: exact MTP${SPECULATIVE_TOKENS} model passed the bounded direct-P2P workload and tore down healthy"
  else
    echo "VERDICT -> $CAMPAIGN_LABEL FAIL: do not advance to the next direct-P2P load gate"
  fi
} | tee -a "$RESULT_DIR/verdict.txt"

trap - EXIT INT TERM
cleanup
if [ "$transaction_rc" -ne 0 ] || [ "$p2p_log_rc" -ne 0 ] || \
   [ "$recovery_rc" -ne 0 ] || [ "$post_card_rc" -ne 0 ] || \
   [ "$post_p2p0_rc" -ne 0 ]; then
  exit 1
fi
