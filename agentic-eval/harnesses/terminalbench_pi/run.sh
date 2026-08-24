#!/usr/bin/env bash
# Run Pi against a live local endpoint on Terminal-Bench 3.0.
# GPU discipline: invoke this through ./bin/gpu-run while the endpoint is live.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/arms.sh"

ARM="${1:?usage: run.sh <arm> <install|smoke|gate|full>}"
SUBSET="${2:-smoke}"
tb_set_arm "$ARM"

HARBOR="$HERE/.venv/bin/harbor"
[ -x "$HARBOR" ] || { echo "run: missing Harbor; run setup.sh first" >&2; exit 1; }

PI_VERSION="${PI_VERSION:-0.84.3}"
TB_DATASET="${TB_DATASET:-terminal-bench/terminal-bench@sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3}"
TB_BASE_URL="${TB_BASE_URL:-http://192.168.10.5:18080/v1}"
TB_HOST="${TB_HOST:-192.168.10.5}"
TB_TIMEOUT_MULTIPLIER="${TB_TIMEOUT_MULTIPLIER:-1.0}"
TB_THINKING="${TB_THINKING:-medium}"

case "$SUBSET" in
  install) N_TASKS="${TB_N_TASKS:-3}"; INSTALL_ONLY=1 ;;
  smoke)   N_TASKS="${TB_N_TASKS:-3}"; INSTALL_ONLY=0 ;;
  gate)    N_TASKS="${TB_N_TASKS:-12}"; INSTALL_ONLY=0 ;;
  full)    N_TASKS="${TB_N_TASKS:-70}"; INSTALL_ONLY=0 ;;
  *) echo "unknown subset '$SUBSET'; valid: install smoke gate full" >&2; exit 2 ;;
esac

if [ "$INSTALL_ONLY" = 0 ]; then
  GOT="$(curl -fsS --max-time 10 "$TB_BASE_URL/models" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("\n".join(str(x.get("id", "")) for x in d.get("data", [])))
')"
  if ! printf '%s\n' "$GOT" | grep -Fxq "$TB_SERVED"; then
    echo "identity mismatch: arm=$TB_ARM expects '$TB_SERVED', endpoint returned: ${GOT:-none}" >&2
    exit 3
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="${TB_JOB_NAME:-${TB_ARM}-${SUBSET}-${STAMP}}"
JOBS_DIR="${TB_JOBS_DIR:-$HERE/jobs}"
mkdir -p "$JOBS_DIR"

ARGS=(
  run
  --dataset "$TB_DATASET"
  --agent pi
  --model "openai/$TB_SERVED"
  --agent-kwarg "model_api=openai-completions"
  --agent-kwarg "version=$PI_VERSION"
  --agent-kwarg "thinking=$TB_THINKING"
  --agent-env "OPENAI_API_KEY=dummy"
  --agent-env "OPENAI_BASE_URL=$TB_BASE_URL"
  --allow-agent-host "$TB_HOST"
  --n-concurrent 1
  --n-tasks "$N_TASKS"
  --max-retries 1
  --timeout-multiplier "$TB_TIMEOUT_MULTIPLIER"
  --job-name "$JOB_NAME"
  --jobs-dir "$JOBS_DIR"
  --yes
)
for task in exam-pdf-eval fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader; do
  ARGS+=(--exclude-task-name "$task")
done
[ "$INSTALL_ONLY" = 1 ] && ARGS+=(--install-only)

echo "config -> arm=$TB_ARM backend=$TB_BACKEND scheme=$TB_SCHEME served=$TB_SERVED context=$TB_CONTEXT"
echo "command -> Harbor=$($HARBOR --version) Pi=$PI_VERSION dataset=$TB_DATASET subset=$SUBSET n=$N_TASKS concurrency=1"
"$HARBOR" "${ARGS[@]}"

JOB_DIR="$JOBS_DIR/$JOB_NAME"
if [ "$INSTALL_ONLY" = 0 ] && [ -f "$JOB_DIR/result.json" ]; then
  python3 "$HERE/analyze.py" "$JOB_DIR" --curve-dir "$JOB_DIR/curves"
fi
echo "result -> $JOB_DIR"
