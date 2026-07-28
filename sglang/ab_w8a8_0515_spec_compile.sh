#!/usr/bin/env bash
# Controlled 0.5.15 A/B for re-enabling torch.compile on the two XPU speculative
# metadata helpers disabled by upstream commit 4fffc6448. Caller holds both cards:
#   ./bin/gpu-run bash sglang/ab_w8a8_0515_spec_compile.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE="$REPO/sglang/serve_w8a8_0515.sh"
PORT="${PORT:-30000}"
CTX="${CTX:-8192}"
MAXREQ="${MAXREQ:-4}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
NAME_OFF="${NAME_OFF:-sglang_w8a8_spec_compile_off}"
NAME_ON="${NAME_ON:-sglang_w8a8_spec_compile_on}"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  docker stop -t 30 "$NAME_OFF" "$NAME_ON" >/dev/null 2>&1 || true
  docker rm "$NAME_OFF" "$NAME_ON" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

run_arm() {
  local label="$1"
  local enabled="$2"
  local name="$3"
  local served="$4"

  echo "ARM -> $label spec_compile=$enabled ctx=$CTX radix=0 maxreq=$MAXREQ"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" SPEC_COMPILE="$enabled" \
    PORT="$PORT" NAME="$name" SERVED="$served" bash "$SERVE" start

  local model_json model_id model_len
  model_json="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models")"
  model_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$model_json")"
  model_len="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' <<<"$model_json")"
  echo "IDENTITY[$label] -> id=$model_id max_model_len=$model_len"
  [ "$model_id" = "$served" ] && [ "$model_len" -ge "$CTX" ] || {
    echo "RESULT[$label] -> FAIL: identity"
    return 1
  }

  local compile_hits
  compile_hits="$(docker logs "$name" 2>&1 | rg -c \
    're-enabled torch.compile for XPU speculative metadata' || true)"
  compile_hits="${compile_hits:-0}"
  echo "ENGAGEMENT[$label] -> spec_compile_log_hits=$compile_hits"
  if [ "$enabled" = 1 ] && [ "$compile_hits" -lt 2 ]; then
    echo "RESULT[$label] -> FAIL: compile override missing on one or both TP ranks"
    return 1
  fi
  if [ "$enabled" = 0 ] && [ "$compile_hits" -ne 0 ]; then
    echo "RESULT[$label] -> FAIL: compile override active in control"
    return 1
  fi

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
  docker stop -t 30 "$name" >/dev/null
  docker rm "$name" >/dev/null
  "$REPO/bin/xpu-health"
}

run_arm \
  "sglang-0.5.15-spec-compile-off" \
  0 \
  "$NAME_OFF" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515-spec-compile-off"

run_arm \
  "sglang-0.5.15-spec-compile-on" \
  1 \
  "$NAME_ON" \
  "qwen36-27b-w8a8-gptq-mtp-sgl0515-spec-compile-on"

echo "VERDICT -> both 0.5.15 speculative-compile A/B arms completed"
