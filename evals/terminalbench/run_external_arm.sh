#!/usr/bin/env bash
# Run Terminal-Bench 3 / Pi against an ALREADY-RUNNING OpenAI-compatible endpoint.
# No server lifecycle and no GPU lease here: the caller owns the server, its
# health checks and its teardown. Pi policy is identical for every arm because
# it is fully determined by THINKING / MAX_TOKENS / CONTEXT_WINDOW / prompt.
#
#   BASE_URL=http://192.168.10.5:18080/v1 MODEL=hotschmoe-dd \
#   CONTEXT_WINDOW=65536 ARM_LABEL=armA API_KEY=... \
#     evals/terminalbench/run_external_arm.sh
#
# BASE_URL must be reachable from inside the Docker task containers (use the
# LAN/host IP, not 127.0.0.1). --print-config and --dry-run touch nothing.
set -euo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/flashnext_iq3_ab_20261013}"
JOBS_ROOT="${JOBS_ROOT:-$RESULTS_ROOT/harbor-jobs}"
MANIFEST="${MANIFEST:-$REPO/evals/terminalbench/tb3_ab24_manifest.json}"
BASE_URL="${BASE_URL:-}"
MODEL="${MODEL:-}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-}"
API_KEY="${API_KEY:-EMPTY}"
ARM_LABEL="${ARM_LABEL:-}"
SHARD="${SHARD:-}"                 # optional shard id suffix, e.g. 01 -> tb3-ab24-01
DATASET="${DATASET:-terminal-bench/terminal-bench@3.0.0}"   # e.g. swe-bench/swe-bench-verified with a SWE manifest
TASK_PREFIX="${TASK_PREFIX:-terminal-bench}"                       # Harbor task ids are <prefix>/<name>
PI_VERSION="${PI_VERSION:-1.2.0}"     # Pi installed in each task container via npm
THINKING="${THINKING:-xhigh}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
N_CONCURRENT=1                     # one request at a time, not overridable
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$THINKING" in
  off)
    MAX_TOKENS="${MAX_TOKENS:-8192}"
    PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-$REPO/evals/terminalbench/pi_concise_off_prompt.j2}"
    ;;
  xhigh)
    MAX_TOKENS="${MAX_TOKENS:-16384}"
    PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-$REPO/evals/terminalbench/pi_concise_prompt.j2}"
    ;;
  *)
    echo "THINKING must be off or xhigh, got $THINKING" >&2
    exit 2
    ;;
esac

MODE=run
case "${1:-}" in
  --print-config) MODE=print-config ;;
  --dry-run) MODE=dry-run ;;
  "") ;;
  *) echo "usage: $0 [--print-config|--dry-run]" >&2; exit 2 ;;
esac

for v in BASE_URL MODEL CONTEXT_WINDOW ARM_LABEL; do
  if [ -z "${!v}" ]; then
    echo "$v is required" >&2
    exit 2
  fi
done
case "$BASE_URL" in
  */v1) ;;
  *) echo "BASE_URL must end in /v1, got $BASE_URL" >&2; exit 2 ;;
esac
if [ "$MAX_TOKENS" -ge "$CONTEXT_WINDOW" ]; then
  echo "MAX_TOKENS ($MAX_TOKENS) must be below CONTEXT_WINDOW ($CONTEXT_WINDOW)" >&2
  exit 2
fi
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

# Task list from the pre-registered manifest (optionally one shard).
mapfile -t TASKS < <(python3 - "$MANIFEST" "$SHARD" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
shard = sys.argv[2]
if shard:
    sid = shard if shard.startswith("tb3-") else f"{m['shards'][0]['id'].rsplit('-', 1)[0]}-{shard}"
    names = next(s["tasks"] for s in m["shards"] if s["id"] == sid)
else:
    names = [t["name"] for t in m["tasks"]]
print("\n".join(names))
PY
)
[ "${#TASKS[@]}" -gt 0 ] || { echo "empty task list" >&2; exit 2; }

# Agent host (hostname[:port] stripped) for Harbor's allow list.
AGENT_HOST="$(python3 -c 'import sys, urllib.parse as u; print(u.urlparse(sys.argv[1]).hostname)' "$BASE_URL")"
case "$AGENT_HOST" in
  127.*|localhost|::1)
    echo "WARNING: $AGENT_HOST is not reachable from Docker task containers" >&2
    ;;
esac

JOB_NAME="${JOB_NAME:-${JOB_PREFIX:-tb3-ab24}-${ARM_LABEL}-${STAMP}}"
JOB_DIR="$JOBS_ROOT/$JOB_NAME"
HARBOR_LOG="$RESULTS_ROOT/${JOB_NAME}-harbor.log"
MODELS_JSON="$JOB_DIR/b70_models.json"
IDENTITY_JSON="$JOB_DIR/b70_identity.json"

HARBOR_ARGS=(
  run -d "$DATASET"
  -a evals.terminalbench.harbor_pi:SglangReasoningPi
  -m "openai/$MODEL"
  --ak model_api=openai-completions
  --ak "thinking=$THINKING"
  --ak "version=$PI_VERSION"
  --ak "context_window=$CONTEXT_WINDOW"
  --ak "max_tokens=$MAX_TOKENS"
  --ak "prompt_template_path=$PROMPT_TEMPLATE_PATH"
  --allow-agent-host "$AGENT_HOST"
  --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT"
  -n "$N_CONCURRENT" -k 1
  -o "$JOBS_ROOT" --job-name "$JOB_NAME" --yes
)
for t in "${TASKS[@]}"; do
  case "$t" in */*) HARBOR_ARGS+=( -i "$t" ) ;; *) HARBOR_ARGS+=( -i "$TASK_PREFIX/$t" ) ;; esac   # Harbor task ids are org/name
done

if [ "$MODE" != run ]; then
  printf 'arm=%s\nbase_url=%s\nmodel=%s\ncontext_window=%s\nthinking=%s\nmax_tokens=%s\n' \
    "$ARM_LABEL" "$BASE_URL" "$MODEL" "$CONTEXT_WINDOW" "$THINKING" "$MAX_TOKENS"
  printf 'prompt_template=%s\nagent_timeout_multiplier=%s\nn_concurrent=%s\nagent_host=%s\n' \
    "$PROMPT_TEMPLATE_PATH" "$AGENT_TIMEOUT_MULT" "$N_CONCURRENT" "$AGENT_HOST"
  printf 'manifest=%s\nn_tasks=%s\njob_dir=%s\n' "$MANIFEST" "${#TASKS[@]}" "$JOB_DIR"
  printf 'harbor_args=%s\n' "${HARBOR_ARGS[*]}"
  exit 0
fi

mkdir -p "$JOB_DIR" "$RESULTS_ROOT"

# Identity: exact served-id match on the live endpoint, recorded verbatim.
curl -fsS -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" >"$MODELS_JSON"
python3 - "$MODELS_JSON" "$MODEL" "$IDENTITY_JSON" "$ARM_LABEL" "$BASE_URL" \
  "$CONTEXT_WINDOW" "$THINKING" "$MAX_TOKENS" "$MANIFEST" <<'PY'
import json, sys, time
models_path, want, out, arm, base, ctx, thinking, max_tokens, manifest = sys.argv[1:]
data = json.load(open(models_path))
ids = [m.get("id") for m in data.get("data", [])]
if want not in ids:
    sys.exit(f"served id {want!r} not in /v1/models ids {ids}")
json.dump({
    "arm": arm, "base_url": base, "served_model": want, "served_ids": ids,
    "max_model_len": [m.get("max_model_len") for m in data["data"] if m.get("id") == want],
    "context_window": int(ctx), "thinking": thinking, "max_tokens": int(max_tokens),
    "manifest": manifest, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "kind": "external endpoint; server/KV dtype identity must be recorded by the caller",
}, open(out, "w"), indent=2)
print("identity ok:", want)
PY

START_EPOCH="$(date +%s)"
set +e
PYTHONPATH="$REPO" \
OPENAI_BASE_URL="$BASE_URL" OPENAI_API_KEY="$API_KEY" \
  harbor "${HARBOR_ARGS[@]}" 2>&1 | tee "$HARBOR_LOG"
HARBOR_RC="${PIPESTATUS[0]}"
set -e
END_EPOCH="$(date +%s)"

# Endpoint health after the job (recorded, not fatal): lets endpoint failures
# be separated from task failures.
if curl -fsS -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" >/dev/null 2>&1; then
  ENDPOINT_AFTER=true
else
  ENDPOINT_AFTER=false
fi
printf '{"arm": "%s", "harbor_rc": %s, "wall_seconds": %s, "endpoint_healthy_after": %s}\n' \
  "$ARM_LABEL" "$HARBOR_RC" "$((END_EPOCH - START_EPOCH))" "$ENDPOINT_AFTER" \
  >"$JOB_DIR/b70_external_lifecycle.json"

python3 "$REPO/evals/terminalbench/summarize.py" "$JOB_DIR" || true
exit "$HARBOR_RC"
