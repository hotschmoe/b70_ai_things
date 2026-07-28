#!/usr/bin/env bash
# Current-stack W8A8 qualification for sglang 0.5.15 on both B70 cards.
# The caller must hold both cards for this script's full lifetime:
#   ./bin/gpu-run bash sglang/qualify_w8a8_0515.sh
#
# This is an experiment driver, not a shelf recipe. It starts one server, checks
# identity/coherence/performance/long-context cache behavior, and always removes it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE="$REPO/sglang/serve_w8a8_0515.sh"
NAME="${NAME:-sglang_w8a8_0515_qual}"
PORT="${PORT:-30000}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp}"
MAXLEN="${MAXLEN:-200000}"
BASE="http://127.0.0.1:$PORT/v1"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ "$rc" != 0 ] && docker inspect "$NAME" >/dev/null 2>&1; then
    echo "SERVER-FAILURE -> final 300 log lines"
    docker logs "$NAME" 2>&1 | tail -300
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "CONFIG -> image=sglang-xpu:mtp-0515 tp=2 maxlen=$MAXLEN radix=1"
echo "CONFIG -> served=$SERVED port=$PORT max_running_requests=${MAXREQ:-4} mamba_cache=${MAMBA_CACHE:-auto} mtp_steps=10"
CTX="$MAXLEN" RADIX=1 PORT="$PORT" NAME="$NAME" SERVED="$SERVED" bash "$SERVE" start

MODEL_JSON="$(curl -fsS --max-time 15 "$BASE/models")"
MODEL_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$MODEL_JSON")"
MODEL_LEN="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' <<<"$MODEL_JSON")"
echo "IDENTITY -> id=$MODEL_ID max_model_len=$MODEL_LEN"
[ "$MODEL_ID" = "$SERVED" ] || {
  echo "RESULT -> FAIL: served model id mismatch"
  exit 1
}
[ "$MODEL_LEN" -ge "$MAXLEN" ] || {
  echo "RESULT -> FAIL: max_model_len is below requested context"
  exit 1
}

docker exec "$NAME" python3 -c \
  'import torch,sglang; print("RUNTIME -> torch",torch.__version__,"sglang",sglang.__version__,"xpu_devices",torch.xpu.device_count())'

POOL_TOKENS="$(
  docker logs "$NAME" 2>&1 |
    rg -o 'max_total_num_tokens=[0-9]+' |
    tail -1 |
    cut -d= -f2 || true
)"
echo "CAPACITY -> max_total_num_tokens=${POOL_TOKENS:-unknown}"
if [ -n "${MIN_POOL_TOKENS:-}" ]; then
  [ -n "$POOL_TOKENS" ] && [ "$POOL_TOKENS" -ge "$MIN_POOL_TOKENS" ] || {
    echo "RESULT -> FAIL: token pool is below required $MIN_POOL_TOKENS"
    exit 1
  }
fi

if [ "${RUN_COHERENCE:-1}" = 1 ]; then
  echo "COHERENCE -> 18 mixed prefill/decode streams"
  python3 "$REPO/vllm/gate_concurrent_coherence.py" "$BASE" "$MODEL_ID" 3 6 200
fi

if [ "${RUN_PERF:-1}" = 1 ]; then
  echo "PERF -> native sglang warm c1/c4 plus sustained 2K-token decode"
  bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$MODEL_ID" "$TOK" "w8a8-0515-200k"
fi

if [ "${RUN_PREFILL:-1}" = 1 ]; then
  echo "PREFILL -> unique-prompt cold measurements"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" "$BASE" "$MODEL_ID" 1 8 2048,32768 2
fi

if [ "${RUN_NEEDLE:-1}" = 1 ]; then
  echo "LONGCTX -> cold and warm retrieval near 190K tokens"
  for pass in cold warm; do
    echo "NEEDLE -> $pass"
    needle_out="$(
      PROBE_HOST="http://127.0.0.1:$PORT" \
        NEEDLE_DEPTH="${NEEDLE_DEPTH:-190000}" \
        NEEDLE_MIN_TOKENS="${NEEDLE_MIN_TOKENS:-180000}" \
        NEEDLE_ONLY=1 \
        python3 "$REPO/vllm/nvfp4/kv_gate.py"
    )"
    echo "$needle_out"
    rg -q 'GATE: 1/1 PASS' <<<"$needle_out" || {
      echo "RESULT -> FAIL: $pass long-context retrieval gate"
      exit 1
    }
  done
fi

echo "CACHE -> metrics"
curl -fsS --max-time 15 "http://127.0.0.1:$PORT/metrics" |
  rg -i 'cache_hit|cached_tokens|prefill_cache|uncached' |
  rg -iv 'bucket|^#' | head -20 || true

if docker logs "$NAME" 2>&1 | rg -i \
  'device_lost|out_of_resources|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' >/dev/null; then
  echo "RESULT -> FAIL: fatal or garbage marker found in server log"
  docker logs "$NAME" 2>&1 | rg -i \
    'device_lost|out_of_resources|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' | tail -80
  exit 1
fi

echo "VERDICT -> PASS: sglang 0.5.15 W8A8 TP2 at $MAXLEN context"
