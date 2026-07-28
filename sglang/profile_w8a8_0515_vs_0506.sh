#!/usr/bin/env bash
# Stage-separated profiler A/B for the coherent sglang 0.5.6 and 0.5.15
# Qwen3.6-27B W8A8 serves. The caller must hold both cards:
#   ./bin/gpu-run bash sglang/profile_w8a8_0515_vs_0506.sh
#
# Each arm uses the same 8K, radix-off TP=2/MTP10 configuration. Sglang's
# profile_by_stage mode records one cold ~2K prefill separately from the first
# five decode batches. Traces persist under /mnt/vm_8tb/b70/sgl_cache/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
PORT="${PORT:-30000}"
CTX="${CTX:-8192}"
MAXREQ="${MAXREQ:-4}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PROFILE_ROOT="${PROFILE_ROOT:-profile_sglang_w8a8_ab_$STAMP}"
NAME_056="${NAME_056:-sglang_w8a8_profile_0506}"
NAME_0515="${NAME_0515:-sglang_w8a8_profile_0515}"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  docker stop -t 30 "$NAME_056" "$NAME_0515" >/dev/null 2>&1 || true
  docker rm "$NAME_056" "$NAME_0515" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

check_profile_response() {
  local response="$1"
  if [ -z "$response" ]; then
    echo "PROFILE API -> HTTP success with empty body"
  else
    printf 'PROFILE API -> HTTP success body=%q\n' "${response:0:200}"
  fi
}

run_arm() {
  local label="$1"
  local short="$2"
  local script="$3"
  local name="$4"
  local served="$5"
  local output_dir="/sgl_cache/$PROFILE_ROOT/$short"
  local host_dir="$ROOT/sgl_cache/$PROFILE_ROOT/$short"
  local profile_id="${short}_${STAMP}"

  echo "ARM -> $label script=$script ctx=$CTX radix=0 maxreq=$MAXREQ"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$served" bash "$script" start

  local model_json model_id model_len
  model_json="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models")"
  model_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$model_json")"
  model_len="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' <<<"$model_json")"
  echo "IDENTITY[$label] -> id=$model_id max_model_len=$model_len"
  [ "$model_id" = "$served" ] && [ "$model_len" -ge "$CTX" ] || {
    echo "RESULT[$label] -> FAIL: identity"
    return 1
  }

  docker exec "$name" python3 -c '
from importlib import metadata
import sglang
import torch
direct = metadata.distribution("sgl-kernel").read_text("direct_url.json")
print("RUNTIME -> torch", torch.__version__, "sglang", sglang.__version__)
print("RUNTIME -> sgl-kernel", metadata.version("sgl-kernel"), direct.strip())
print("RUNTIME -> transformers", metadata.version("transformers"))
'

  mkdir -p "$host_dir"
  echo "PROFILE[$label] -> output=$host_dir"
  local profile_response
  profile_response="$(curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
    -H 'content-type: application/json' \
    -d "{
      \"output_dir\":\"$output_dir\",
      \"num_steps\":5,
      \"activities\":[\"CPU\",\"XPU\"],
      \"profile_by_stage\":true,
      \"with_stack\":false,
      \"record_shapes\":true,
      \"profile_id\":\"$profile_id\",
      \"profile_prefix\":\"$short\"
    }")"
  check_profile_response "$profile_response"

  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$model_id" 1 32 2048 1

  local found=0
  for _ in $(seq 1 30); do
    if [ -n "$(find "$host_dir" -maxdepth 1 -type f -name '*.trace.json.gz' -print -quit)" ]; then
      found=1
      break
    fi
    sleep 1
  done
  [ "$found" = 1 ] || {
    echo "RESULT[$label] -> FAIL: no profiler trace in $host_dir"
    return 1
  }

  find "$host_dir" -maxdepth 1 -type f -name '*.trace.json.gz' \
    -printf 'TRACE -> %p %s bytes\n' | sort
  python3 "$REPO/scripts/112_parse_trace.py" "$host_dir"

  if docker logs "$name" 2>&1 | rg -i \
    'device_lost|out_of_resources|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' >/dev/null; then
    echo "RESULT[$label] -> FAIL: fatal or garbage marker in server log"
    return 1
  fi

  echo "RESULT[$label] -> PASS"
  docker stop -t 30 "$name" >/dev/null
  docker rm "$name" >/dev/null
  "$REPO/bin/xpu-health"
}

run_arm \
  "sglang-0.5.6" \
  "sgl056" \
  "$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh" \
  "$NAME_056" \
  "qwen36-27b-w8a8-gptq-mtp-sgl056-profile"

run_arm \
  "sglang-0.5.15" \
  "sgl0515" \
  "$REPO/sglang/serve_w8a8_0515.sh" \
  "$NAME_0515" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515-profile"

echo "VERDICT -> both stage-separated profile arms completed"
echo "TRACES -> $ROOT/sgl_cache/$PROFILE_ROOT"
