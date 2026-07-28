#!/usr/bin/env bash
# Controlled shelf-promotion A/B: sglang 0.5.6 versus 0.5.15 W8A8.
# The caller must hold both cards for this script's full lifetime:
#   ./bin/gpu-run bash sglang/ab_w8a8_0515_vs_0506.sh
#
# Both arms use the same 8K, radix-off, TP=2, MTP10 configuration and the same
# coherence/performance probes. The script always removes its containers.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-30000}"
CTX="${CTX:-8192}"
MAXREQ="${MAXREQ:-4}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
NAME_056="${NAME_056:-sglang_w8a8_ab_0506}"
NAME_0515="${NAME_0515:-sglang_w8a8_ab_0515}"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  docker rm -f "$NAME_056" "$NAME_0515" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

run_arm() {
  local label="$1"
  local script="$2"
  local name="$3"
  local served="$4"

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
  docker exec "$name" python3 -c \
    'import torch,sglang; print("RUNTIME -> torch",torch.__version__,"sglang",sglang.__version__,"xpu_devices",torch.xpu.device_count())'

  echo "COHERENCE[$label] -> 18 mixed prefill/decode streams"
  python3 "$REPO/vllm/gate_concurrent_coherence.py" \
    "http://127.0.0.1:$PORT/v1" "$model_id" 3 6 200

  echo "PERF[$label] -> native c1/c4 and sustained decode"
  bash "$REPO/sglang/perf_regime.sh" \
    "$name" "$PORT" "$model_id" "$TOK" "$label"

  echo "PREFILL[$label] -> unique cold 2K"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$model_id" 1 8 2048 3

  echo "CODE[$label] -> c1 then c4"
  python3 "$REPO/vllm/nvfp4/bench_code.py" \
    "http://127.0.0.1:$PORT/v1" "$model_id" 1 256 3
  python3 "$REPO/vllm/nvfp4/bench_code.py" \
    "http://127.0.0.1:$PORT/v1" "$model_id" 4 256 2

  if docker logs "$name" 2>&1 | rg -i \
    'device_lost|out_of_resources|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' >/dev/null; then
    echo "RESULT[$label] -> FAIL: fatal or garbage marker in server log"
    return 1
  fi

  echo "RESULT[$label] -> PASS"
  docker rm -f "$name" >/dev/null
  "$REPO/bin/xpu-health"
}

run_arm \
  "sglang-0.5.6" \
  "$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh" \
  "$NAME_056" \
  "qwen36-27b-w8a8-gptq-mtp-sgl056"

run_arm \
  "sglang-0.5.15" \
  "$REPO/sglang/serve_w8a8_0515.sh" \
  "$NAME_0515" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515"

echo "VERDICT -> both controlled A/B arms completed"
