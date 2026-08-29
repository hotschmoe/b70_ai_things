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
MAX_TOKENS="${MAX_TOKENS-}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH-}"
XHIGH_THINKCAP="${XHIGH_THINKCAP:-4096}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$THINKING" in
  off)
    MAX_TOKENS="${MAX_TOKENS:-8192}"
    PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-$REPO/evals/terminalbench/pi_concise_off_prompt.j2}"
    POLICY_THINKCAP=""
    ;;
  xhigh)
    MAX_TOKENS="${MAX_TOKENS:-16384}"
    PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-$REPO/evals/terminalbench/pi_concise_prompt.j2}"
    POLICY_THINKCAP="$XHIGH_THINKCAP"
    ;;
  *)
    echo "THINKING must be off or xhigh, got $THINKING" >&2
    exit 2
    ;;
esac

if [ "$PREFILL_WINDOW" -gt "$CONTEXT_WINDOW" ]; then
  PREFILL_WINDOW="$CONTEXT_WINDOW"
fi

MODE=run
if [ "${1:-}" = --print-config ]; then
  MODE=print-config
  shift
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

if [ "$MODE" = run ] && [ "$LEASED" = 0 ]; then
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
      "GRAPH_BS=1" "TOOLPARSER=qwen3_coder"
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
      "SERVED=$SERVED"
    )
    ;;
  qwen-nvfp4)
    LAUNCHER="$REPO/sglang/nvfp4/serve_qwen38_nvfp4_refresh.sh"
    CONTAINER="tb3_qwen38_nvfp4"
    SERVED="qwen3.8-27b-NVFP4-radixark-full-tp2-bf16kv-tb3"
    START_ENV=(
      "NAME=$CONTAINER" "PORT=$PORT" "CTX=$CONTEXT_WINDOW"
      "MEMFRAC=0.70" "MAXREQ=1" "DECODE_GRAPH=full" "GRAPH_BS=1"
      "TOOLPARSER=qwen3_coder" "SERVED=$SERVED"
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
      "TOOLPARSER=qwen3_coder" "SERVED=$SERVED"
    )
    ;;
esac

case "$ARM" in
  qwen-w8a8|qwen-w8a8-reclaim500|qwen-nvfp4|ornith-w8a8)
    START_ENV+=("THINKCAP=$POLICY_THINKCAP")
    ;;
esac

if [ "$MODE" = print-config ]; then
  printf 'arm=%s\nthinking=%s\nmax_tokens=%s\nprompt_template=%s\nthinkcap=%s\n' \
    "$ARM" "$THINKING" "$MAX_TOKENS" "$PROMPT_TEMPLATE_PATH" \
    "${POLICY_THINKCAP:-}"
  printf 'start_env=%s\n' "${START_ENV[*]}"
  exit 0
fi

JOB_NAME="${JOB_NAME:-tb3-${ARM}-${STAMP}}"
JOB_DIR="$JOBS_ROOT/$JOB_NAME"
SERVER_LOG="$ROOT/evals/${JOB_NAME}-server.log"
HARBOR_LOG="$ROOT/evals/${JOB_NAME}-harbor.log"
TIMING_JSON="$JOB_DIR/b70_lifecycle.json"
IDENTITY_JSON="$JOB_DIR/b70_identity.json"
MODELS_JSON="$JOB_DIR/b70_models.json"
MACHINE_START_EPOCH="$(date +%s)"
PREHEALTH_END_EPOCH=""
SERVER_START_EPOCH=""
READY_EPOCH=""
HARBOR_END_EPOCH=""
PRETEARDOWN_CHECK_EPOCH=""
TEARDOWN_END_EPOCH=""
PRE_CARD_HEALTH=false
PRE_COLLECTIVE_HEALTH=false

write_timing() {
  local posthealth_epoch="$1" exit_code="$2" post_card="$3" post_collective="$4"
  local endpoint_healthy="$5" endpoint_down="$6" fatal_markers="$7"
  mkdir -p "$JOB_DIR"
  PYTHONPATH="$REPO" python3 "$REPO/evals/terminalbench/campaign_evidence.py" \
    lifecycle --output "$TIMING_JSON" --arm "$ARM" --served-model "$SERVED" \
    --exit-code "$exit_code" --machine-start "$MACHINE_START_EPOCH" \
    --prehealth-end "$PREHEALTH_END_EPOCH" --server-start "$SERVER_START_EPOCH" \
    --server-ready "$READY_EPOCH" --harbor-end "$HARBOR_END_EPOCH" \
    --preteardown-check "$PRETEARDOWN_CHECK_EPOCH" \
    --teardown-end "$TEARDOWN_END_EPOCH" --posthealth-end "$posthealth_epoch" \
    --endpoint-healthy-before-teardown "$endpoint_healthy" \
    --endpoint-down-after-teardown "$endpoint_down" \
    --pre-card-health "$PRE_CARD_HEALTH" \
    --pre-collective-health "$PRE_COLLECTIVE_HEALTH" \
    --post-card-health "$post_card" --post-collective-health "$post_collective" \
    --fatal-server-markers "$fatal_markers"
}

cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  local endpoint_healthy=null endpoint_down=null
  local post_card=false post_collective=false
  local fatal_markers='[]'
  PRETEARDOWN_CHECK_EPOCH="$(date +%s)"
  if [ -n "$READY_EPOCH" ]; then
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
      endpoint_healthy=true
    else
      endpoint_healthy=false
      [ "$rc" -ne 0 ] || rc=1
    fi
  fi
  docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1 || true
  fatal_markers="$(python3 - "$SERVER_LOG" <<'PY'
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace") if Path(sys.argv[1]).is_file() else ""
patterns = (
    r"ZE_RESULT_ERROR_DEVICE_LOST",
    r"linear_stream\.h:90",
    r"GPU (?:virtual-memory|VM) fault",
    r"engine core[^\n]*(?:died|dead)",
    r"(?:out of memory|Killed process|std::bad_alloc)",
    r"\bNaN\b",
)
found = sorted({match.group(0)[:240] for pattern in patterns for match in re.finditer(pattern, text, re.IGNORECASE)})
print(json.dumps(found, ensure_ascii=True))
PY
)"
  [ "$fatal_markers" = '[]' ] || { [ "$rc" -ne 0 ] || rc=1; }
  if [ -n "$SERVER_START_EPOCH" ]; then
    env "${START_ENV[@]}" bash "$LAUNCHER" stop || { [ "$rc" -ne 0 ] || rc=1; }
    TEARDOWN_END_EPOCH="$(date +%s)"
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
      endpoint_down=false
      [ "$rc" -ne 0 ] || rc=1
    else
      endpoint_down=true
    fi
  fi
  "$REPO/bin/xpu-health" && post_card=true || { [ "$rc" -ne 0 ] || rc=1; }
  "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180 && post_collective=true || { [ "$rc" -ne 0 ] || rc=1; }
  write_timing "$(date +%s)" "$rc" "$post_card" "$post_collective" \
    "$endpoint_healthy" "$endpoint_down" "$fatal_markers" || { [ "$rc" -ne 0 ] || rc=1; }
  echo "lifecycle -> $TIMING_JSON"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$ROOT/evals" "$JOBS_ROOT" "$JOB_DIR"
if "$REPO/bin/xpu-health"; then
  PRE_CARD_HEALTH=true
else
  exit 1
fi
if "$REPO/bin/xpu-collective-health" --p2p 0 --timeout 180; then
  PRE_COLLECTIVE_HEALTH=true
else
  exit 1
fi
PREHEALTH_END_EPOCH="$(date +%s)"

SERVER_START_EPOCH="$(date +%s)"
env "${START_ENV[@]}" "LOG=$SERVER_LOG" bash "$LAUNCHER" start
READY_EPOCH="$(date +%s)"
docker logs "$CONTAINER" >"$SERVER_LOG" 2>&1
curl -fsS "http://localhost:$PORT/v1/models" >"$MODELS_JSON"
PYTHONPATH="$REPO" python3 "$REPO/evals/terminalbench/campaign_evidence.py" \
  identity --models-json "$MODELS_JSON" --server-log "$SERVER_LOG" \
  --expected-model "$SERVED" --expected-target-dtype bfloat16 \
  --expected-kv-dtype bfloat16 --output "$IDENTITY_JSON"

HARBOR_ARGS=(
  run -d terminal-bench/terminal-bench@3.0.0
  -a evals.terminalbench.harbor_pi:SglangReasoningPi
  -m "openai/$SERVED"
  --ak model_api=openai-completions
  --ak "thinking=$THINKING"
  --ak version=0.84.3
  --ak "context_window=$CONTEXT_WINDOW"
  --ak "max_tokens=$MAX_TOKENS"
  --ak "prompt_template_path=$PROMPT_TEMPLATE_PATH"
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
