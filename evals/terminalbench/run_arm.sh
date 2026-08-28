#!/usr/bin/env bash
# Run one BF16-KV Terminal-Bench 3.0.0 arm from server start through teardown.
set -euo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
JOBS_ROOT="${JOBS_ROOT:-$ROOT/evals/harbor-jobs}"
MODEL_HOST="${MODEL_HOST:-192.168.10.5}"
PORT="${PORT:-18080}"
INCLUDE_TASK="${INCLUDE_TASK:-}"
N_TASKS="${N_TASKS:-}"
N_CONCURRENT="${N_CONCURRENT:-1}"
THINKING="${THINKING:-xhigh}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-65536}"
PREFILL_WINDOW="${PREFILL_WINDOW:-16384}"
GPTQ_UTIL="${GPTQ_UTIL:-0.90}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [ "$PREFILL_WINDOW" -gt "$CONTEXT_WINDOW" ]; then
  PREFILL_WINDOW="$CONTEXT_WINDOW"
fi

if [ "${1:-}" = --leased ]; then
  LEASED=1
  shift
else
  LEASED=0
fi
ARM="${1:-}"
case "$ARM" in
  qwen-w8a8|qwen-w8a8-reclaim500|qwen-nvfp4|qwen-gptq-int4|ornith-w8a8) ;;
  *)
    echo "usage: $0 {qwen-w8a8|qwen-w8a8-reclaim500|qwen-nvfp4|qwen-gptq-int4|ornith-w8a8}" >&2
    exit 2
    ;;
esac

if [ "$LEASED" = 0 ]; then
  exec env B70_AGENT="terminalbench-$ARM" "$REPO/bin/gpu-run" \
    bash "$0" --leased "$ARM"
fi

case "$ARM" in
  qwen-w8a8)
    LAUNCHER="$REPO/sglang/w8a8/serve_qwen38_w8a8.sh"
    CONTAINER="tb3_qwen38_w8a8"
    SERVED="qwen3.8-27b-W8A8-gptq-gdn-rtn-full-tp2-bf16kv-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "CTX=$CONTEXT_WINDOW"
      "MEMFRAC=0.70" "MAXREQ=1" "MTP=0" "DECODE_GRAPH=full"
      "GRAPH_BS=1" "TOOLPARSER=qwen3_coder" "THINKCAP=4096"
      "SERVED=$SERVED"
    )
    ;;
  qwen-w8a8-reclaim500)
    LAUNCHER="$REPO/sglang/w8a8/serve_qwen38_w8a8.sh"
    CONTAINER="tb3_qwen38_w8a8_reclaim500"
    SERVED="qwen3.8-27b-W8A8-gptq-gdn-rtn-breakable-reclaim500-tp2-bf16kv-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "CTX=$CONTEXT_WINDOW"
      "MEMFRAC=0.70" "MAXREQ=1" "MTP=0" "DECODE_GRAPH=breakable"
      "GRAPH_BS=1" "CG_RECLAIM=500" "TOOLPARSER=qwen3_coder"
      "THINKCAP=4096" "SERVED=$SERVED"
    )
    ;;
  qwen-nvfp4)
    LAUNCHER="$REPO/sglang/nvfp4/serve_qwen38_nvfp4_refresh.sh"
    CONTAINER="tb3_qwen38_nvfp4"
    SERVED="qwen3.8-27b-NVFP4-radixark-full-tp2-bf16kv-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "CTX=$CONTEXT_WINDOW"
      "MEMFRAC=0.70" "MAXREQ=1" "DECODE_GRAPH=full" "GRAPH_BS=1"
      "TOOLPARSER=qwen3_coder" "THINKCAP=4096" "SERVED=$SERVED"
    )
    ;;
  qwen-gptq-int4)
    LAUNCHER="$REPO/vllm/gptq_int4/serve_qwen38_gptq_int4_v0272.sh"
    CONTAINER="tb3_qwen38_gptq_int4_v0272"
    SERVED="qwen3.8-27b-GPTQ-INT4-g128-mtp4-draft-lmhead-int4-bf16kv-vllm0272-tp1-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "DEVICE=0" "MAXLEN=$CONTEXT_WINDOW"
      "MAXSEQS=1" "MAXBATCH=$PREFILL_WINDOW" "UTIL=$GPTQ_UTIL" "MTPTOK=4"
      "DRAFT_LMHEAD_INT4=1" "TOOLPARSER=qwen3_coder" "SERVED=$SERVED"
      "B70_LOGDIR=$ROOT/evals"
    )
    ;;
  ornith-w8a8)
    LAUNCHER="$REPO/sglang/w8a8/serve_ornith15_w8a8_refresh.sh"
    CONTAINER="tb3_ornith15_w8a8_reclaim500"
    SERVED="ornith-1.5-35b-a3b-W8A8-rtn-target-breakable-reclaim500-bf16kv-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "CTX=$CONTEXT_WINDOW"
      "MEMFRAC=0.70" "MAXREQ=1" "MTP=0" "DENSE_NATIVE=0"
      "DECODE_GRAPH=breakable" "GRAPH_BS=1" "CG_RECLAIM=500"
      "TOOLPARSER=qwen3_coder" "THINKCAP=4096" "SERVED=$SERVED"
    )
    ;;
esac

JOB_NAME="${JOB_NAME:-tb3-${ARM}-${STAMP}}"
JOB_DIR="$JOBS_ROOT/$JOB_NAME"
SERVER_LOG="$ROOT/evals/${JOB_NAME}-server.log"
HARBOR_LOG="$ROOT/evals/${JOB_NAME}-harbor.log"
TIMING_JSON="$JOB_DIR/b70_lifecycle.json"
START_EPOCH=""
READY_EPOCH=""
HARBOR_END_EPOCH=""

write_timing() {
  local teardown_epoch="$1" exit_code="$2"
  mkdir -p "$JOB_DIR"
  python3 - "$TIMING_JSON" "$ARM" "$SERVED" "$START_EPOCH" \
    "$READY_EPOCH" "$HARBOR_END_EPOCH" "$teardown_epoch" "$exit_code" <<'PY'
import json
import sys
from pathlib import Path

path, arm, served, start, ready, harbor_end, teardown_end, exit_code = sys.argv[1:]

def number(value):
    return int(value) if value else None

values = {
    "server_start_epoch": number(start),
    "server_ready_epoch": number(ready),
    "harbor_end_epoch": number(harbor_end),
    "teardown_end_epoch": number(teardown_end),
}
data = {
    "arm": arm,
    "served_model": served,
    "bf16_kv": True,
    "exit_code": int(exit_code),
    **values,
}
if values["server_start_epoch"] is not None and values["server_ready_epoch"] is not None:
    data["startup_seconds"] = values["server_ready_epoch"] - values["server_start_epoch"]
if values["server_ready_epoch"] is not None and values["harbor_end_epoch"] is not None:
    data["harbor_seconds"] = values["harbor_end_epoch"] - values["server_ready_epoch"]
if values["server_start_epoch"] is not None and values["harbor_end_epoch"] is not None:
    data["start_through_harbor_seconds"] = values["harbor_end_epoch"] - values["server_start_epoch"]
if values["server_start_epoch"] is not None and values["teardown_end_epoch"] is not None:
    data["end_to_end_seconds"] = values["teardown_end_epoch"] - values["server_start_epoch"]
Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
}

cleanup() {
  local rc=$?
  set +e
  [ -n "$HARBOR_END_EPOCH" ] || HARBOR_END_EPOCH="$(date +%s)"
  docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1 || true
  env "${START_ENV[@]}" bash "$LAUNCHER" stop || true
  "$REPO/bin/xpu-health" || true
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 || true
  write_timing "$(date +%s)" "$rc"
  echo "lifecycle -> $TIMING_JSON"
  return "$rc"
}
trap cleanup EXIT INT TERM

mkdir -p "$ROOT/evals" "$JOBS_ROOT"
"$REPO/bin/xpu-health"
"$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180

START_EPOCH="$(date +%s)"
env "${START_ENV[@]}" "LOG=$SERVER_LOG" bash "$LAUNCHER" start
READY_EPOCH="$(date +%s)"

HARBOR_ARGS=(
  run -d terminal-bench/terminal-bench@3.0.0
  -a evals.terminalbench.harbor_pi:SglangReasoningPi
  -m "openai/$SERVED"
  --ak model_api=openai-completions
  --ak "thinking=$THINKING"
  --ak version=0.84.3
  --ak "context_window=$CONTEXT_WINDOW"
  --ak "max_tokens=$MAX_TOKENS"
  --ak "prompt_template_path=$REPO/evals/terminalbench/pi_concise_prompt.j2"
  --allow-agent-host "$MODEL_HOST"
  -n "$N_CONCURRENT" -k 1
  -o "$JOBS_ROOT" --job-name "$JOB_NAME" --yes
)
if [ -n "$INCLUDE_TASK" ]; then
  HARBOR_ARGS+=( -i "$INCLUDE_TASK" )
fi
if [ -n "$N_TASKS" ]; then
  HARBOR_ARGS+=( -l "$N_TASKS" )
fi

set +e
PYTHONPATH="$REPO" \
OPENAI_BASE_URL="http://$MODEL_HOST:$PORT/v1" OPENAI_API_KEY=EMPTY \
  harbor "${HARBOR_ARGS[@]}" 2>&1 | tee "$HARBOR_LOG"
HARBOR_RC="${PIPESTATUS[0]}"
set -e
HARBOR_END_EPOCH="$(date +%s)"

python3 "$REPO/evals/terminalbench/summarize.py" "$JOB_DIR" || true
exit "$HARBOR_RC"
