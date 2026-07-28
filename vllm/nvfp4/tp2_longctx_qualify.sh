#!/usr/bin/env bash
# Current-stack qualification for one-request long context on NVFP4 TP=2.
# The caller must hold both cards:
#   ./bin/gpu-run bash vllm/nvfp4/tp2_longctx_qualify.sh
#
# This is an experiment driver, not a shelf recipe. It starts one server, runs
# identity/coherence/performance/needle/soak gates, and always tears it down.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh"
PORT="${PORT:-18079}"
NAME="${NAME:-nvfp4_tp2_longctx_qual}"
SERVED_FORCE="${SERVED_FORCE:-qwen3.6-27b-NVFP4-modelopt-fused-graph-mtp5}"
BASE="http://127.0.0.1:$PORT/v1"

export TP=2
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export MAXLEN="${MAXLEN:-200000}"
export UTIL="${UTIL:-0.85}"
export MAXSEQS="${MAXSEQS:-8}"
export MAXBATCH="${MAXBATCH:-16384}"
export CAPSIZES="${CAPSIZES:-1,2,4,8}"
export MTPTOK="${MTPTOK:-5}"
export PREFIXCACHE="${PREFIXCACHE:-1}"
export PUSH_AR="${PUSH_AR:-1}"
export PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-1}"
export PUSH_AR_MAXB="${PUSH_AR_MAXB:-268435456}"
export KV_FP8="${KV_FP8:-1}"
export KV_SCALES="${KV_SCALES:-$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json}"
export FUSED_SO="${FUSED_SO:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so}"
export GDN_LIB="${GDN_LIB:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
export PORT NAME SERVED_FORCE
export TOOLCALL="${TOOLCALL:-1}"
export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"
export B70_EXTRA_ENV="${B70_EXTRA_ENV:-B70_PC_EAGLE_KEEP=1 B70_PC_CHUNK_ALIGN=1 B70_NVFP4_F8_SCALE_M_MAX=8}"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ "$rc" != 0 ] && docker inspect "$NAME" >/dev/null 2>&1; then
    echo "SERVER-FAILURE -> final 400 log lines"
    docker logs "$NAME" 2>&1 | tail -400
  fi
  bash "$SHELF" stop >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "CONFIG -> IMG=$IMG TP=$TP MAXLEN=$MAXLEN KV_FP8=$KV_FP8 MTP=$MTPTOK"
echo "CONFIG -> prefix=$PREFIXCACHE push_ar=$PUSH_AR graph_push=$PUSH_AR_GRAPH"
bash "$SHELF" start

echo -n "WAIT -> health"
healthy=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
    healthy=1
    break
  fi
  echo -n "."
  sleep 5
done
echo
[ "$healthy" = 1 ] || {
  echo "RESULT -> FAIL: server did not become healthy"
  docker logs "$NAME" 2>&1 | tail -120
  exit 1
}

MODEL_JSON="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models")"
MODEL_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$MODEL_JSON")"
MODEL_LEN="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' <<<"$MODEL_JSON")"
echo "IDENTITY -> id=$MODEL_ID max_model_len=$MODEL_LEN"
[ "$MODEL_ID" = "$SERVED_FORCE" ] || {
  echo "RESULT -> FAIL: served model id mismatch"
  exit 1
}
[ "$MODEL_LEN" -ge "$MAXLEN" ] || {
  echo "RESULT -> FAIL: max_model_len is below requested context"
  exit 1
}
KV_INJECTS="$(docker logs "$NAME" 2>&1 | rg -c 'injected KV scales' || true)"
echo "KV-SCALES -> injected lines=$KV_INJECTS (expect 16 layers x 2 ranks)"
[ "$KV_INJECTS" -ge 32 ] || {
  echo "RESULT -> FAIL: calibrated KV scales were not injected on both ranks"
  exit 1
}

python3 "$REPO/vllm/gate_concurrent_coherence.py" "$BASE" "$MODEL_ID" 3 6 200
if [ "${RUN_STRESS:-1}" = 1 ]; then
  for rep in $(seq 1 "${STRESS_REPS:-2}"); do
    echo "STRESS -> 36 streams rep $rep/${STRESS_REPS:-2}"
    python3 "$REPO/vllm/gate_concurrent_coherence.py" "$BASE" "$MODEL_ID" 6 6 200
  done
fi

if [ "${RUN_PERF:-1}" = 1 ]; then
  python3 "$REPO/vllm/nvfp4/bench_code.py" "$BASE" "$MODEL_ID" 1 256 3
  python3 "$REPO/vllm/nvfp4/bench_code.py" "$BASE" "$MODEL_ID" 4 256 2
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" "$BASE" "$MODEL_ID" 1 8 2048,32768 2
fi
METRICS="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/metrics")"
ACCEPTED="$(awk '/vllm:spec_decode_num_accepted_tokens_total/{v=$NF} END{print v+0}' <<<"$METRICS")"
DRAFTS="$(awk '/vllm:spec_decode_num_drafts_total/{v=$NF} END{print v+0}' <<<"$METRICS")"
DRAFT_TOK="$(awk '/vllm:spec_decode_num_draft_tokens_total/{v=$NF} END{print v+0}' <<<"$METRICS")"
awk -v a="$ACCEPTED" -v d="$DRAFTS" -v dt="$DRAFT_TOK" \
  'BEGIN {
     if (d <= 0 || dt <= 0) {
       print "MTP -> FAIL: no acceptance telemetry";
       exit 1
     }
     printf "MTP -> accepted=%.0f drafts=%.0f draft_tok=%.0f accept_len=%.3f accept_rate=%.3f\n",
       a, d, dt, 1 + a / d, a / dt
   }'

if [ "${LOCALARGMAX_SHADOW_GATE:-0}" = 1 ]; then
  SERVER_LOG="$(docker logs "$NAME" 2>&1)"
  SHADOW_LINES="$(rg -c '\[localargmax-shadow\]' <<<"$SERVER_LOG" || true)"
  GLOBAL_BAD="$(rg -c 'global_mismatch=[1-9][0-9]*/' <<<"$SERVER_LOG" || true)"
  CPU_IDX_BAD="$(rg -c 'cpu_idx=[1-9][0-9]*' <<<"$SERVER_LOG" || true)"
  CPU_VALUE_BAD="$(rg -c 'cpu_value=[1-9][0-9]*/' <<<"$SERVER_LOG" || true)"
  SHADOW_LINES="${SHADOW_LINES:-0}"
  GLOBAL_BAD="${GLOBAL_BAD:-0}"
  CPU_IDX_BAD="${CPU_IDX_BAD:-0}"
  CPU_VALUE_BAD="${CPU_VALUE_BAD:-0}"
  echo "LOCALARGMAX-SHADOW -> lines=$SHADOW_LINES global_bad=$GLOBAL_BAD cpu_idx_bad=$CPU_IDX_BAD cpu_value_bad=$CPU_VALUE_BAD"
  [ "$SHADOW_LINES" -ge "${LOCALARGMAX_SHADOW_MIN:-64}" ] || {
    echo "RESULT -> FAIL: insufficient local-argmax shadow comparisons"
    exit 1
  }
  [ "$GLOBAL_BAD" = 0 ] && [ "$CPU_IDX_BAD" = 0 ] && [ "$CPU_VALUE_BAD" = 0 ] || {
    echo "RESULT -> FAIL: local-argmax shadow mismatch"
    rg '\[localargmax-(verify|shadow)\]' <<<"$SERVER_LOG" | tail -120
    exit 1
  }
fi

if [ "${REPLICATED_HEAD_SHADOW_GATE:-0}" = 1 ]; then
  SERVER_LOG="$(docker logs "$NAME" 2>&1)"
  READY_LINES="$(rg -c '\[replicated-head\] ready' <<<"$SERVER_LOG" || true)"
  SHADOW_LINES="$(rg -c '\[replicated-head-shadow\]' <<<"$SERVER_LOG" || true)"
  GLOBAL_BAD="$(rg -c 'global_mismatch=[1-9][0-9]*/' \
    <<<"$(rg '\[replicated-head-shadow\]' <<<"$SERVER_LOG" || true)" || true)"
  READY_LINES="${READY_LINES:-0}"
  SHADOW_LINES="${SHADOW_LINES:-0}"
  GLOBAL_BAD="${GLOBAL_BAD:-0}"
  echo "REPLICATED-HEAD-SHADOW -> ready=$READY_LINES lines=$SHADOW_LINES global_bad=$GLOBAL_BAD"
  if [[ "${B70_EXTRA_ENV:-}" == *"LOCALARGMAX_REPLICATED_DEBUG=1"* ]]; then
    rg '\[replicated-head-(hash|mismatch)\]' <<<"$SERVER_LOG" || true
  fi
  [ "$READY_LINES" -ge 2 ] || {
    echo "RESULT -> FAIL: replicated head was not initialized on both TP ranks"
    exit 1
  }
  [ "$SHADOW_LINES" -ge "${REPLICATED_HEAD_SHADOW_MIN:-64}" ] || {
    echo "RESULT -> FAIL: insufficient replicated-head shadow comparisons"
    exit 1
  }
  [ "$GLOBAL_BAD" = 0 ] || {
    echo "RESULT -> FAIL: replicated-head shadow mismatch"
    rg '\[replicated-head(-shadow)?\]' <<<"$SERVER_LOG" | tail -120
    exit 1
  }
fi

if [ "${RUN_NEEDLE:-1}" = 1 ]; then
  for pass in cold warm; do
    echo "NEEDLE -> $pass"
    PROBE_HOST="http://127.0.0.1:$PORT" \
      NEEDLE_DEPTH="${NEEDLE_DEPTH:-190000}" \
      NEEDLE_MIN_TOKENS="${NEEDLE_MIN_TOKENS:-180000}" \
      python3 "$REPO/vllm/nvfp4/kv_gate.py"
  done
fi

if [ "${RUN_SOAK:-1}" = 1 ]; then
  PROBE_HOST=127.0.0.1 PORT="$PORT" WORKERS="${SOAK_WORKERS:-4}" \
    CTX_CHARS="${SOAK_CTX_CHARS:-28000}" MAXTOK="${SOAK_MAXTOK:-4000}" \
    CEIL_TOK="${SOAK_CEIL_TOK:-40000}" CEIL_SEC="${SOAK_CEIL_SEC:-1800}" \
    MIN_PER_WORKER="${SOAK_MIN_PER_WORKER:-8000}" \
    python3 "$REPO/vllm/soak_concurrent.py" "$SERVED_FORCE"
fi

if docker logs "$NAME" 2>&1 | rg -i \
  'device_lost|out_of_resources|enginedead|linear_stream|!!!!|(^|[^a-z])nan([^a-z]|$)' >/dev/null; then
  echo "RESULT -> FAIL: fatal/garbage marker found in server log"
  docker logs "$NAME" 2>&1 | rg -i \
    'device_lost|out_of_resources|enginedead|linear_stream|!!!!|(^|[^a-z])nan([^a-z]|$)' | tail -80
  exit 1
fi

echo "VERDICT -> PASS: current-stack NVFP4 TP=2 long-context qualification"
