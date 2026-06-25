#!/usr/bin/env bash
# agentic-eval/harnesses/tau2/run.sh -- run tau2-bench (retail domain) against the served model.
#
#   bash run.sh <CONFIG_LABEL> <SUBSET>     e.g.  bash run.sh 27b-int4 standard
#
# tau2-bench is a MULTI-TURN tool-use benchmark with a SEPARATE user-simulator LLM. For a clean
# quant A/B the user-sim MUST be a FIXED model held constant across all four configs; it must NOT be
# the model-under-test (that would change both sides of the conversation when the quant changes ->
# confounded). The user-sim is configured via env (USER_SIM_MODEL / USER_SIM_BASE_URL /
# USER_SIM_API_KEY). If USER_SIM_MODEL is unset, this run SKIPS cleanly (emits a score=null result
# so the scoreboard shows "not run" rather than crashing). See README.md.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$HERE/../.." && pwd)/lib/common.sh"
ae_set_config "$1"
SUBSET="${2:-standard}"

VENV="$HERE/.venv"
TAU2_BIN="$VENV/bin/tau2"
SRC_DIR="$HERE/src/tau2-bench"
export TAU2_DATA_DIR="$SRC_DIR/data"          # tasks/db/policy AND simulations/ live under here
RUNS_DIR="$HERE/runs"; mkdir -p "$RUNS_DIR"
PARSED="$RESULTS_DIR/.tau2.parsed.json"
OUT="$RESULTS_DIR/tau2.json"

emit() {  # emit <parsed.json> <tok-before> <tok-after> <start> <end>
  "$AE_PY" "$AE_LIB/evallib.py" emit \
    --config "$EVAL_LABEL" --harness tau2 --subset "$SUBSET" --served "$EVAL_SERVED" \
    --parsed "$1" --tok-before "$2" --tok-after "$3" --start "$4" --end "$5" \
    --out "$OUT" --meta "temperature=$AE_TEMPERATURE" --meta "user_sim=${USER_SIM_MODEL:-NONE}"
}

# ---- SKIP path: no fixed user-sim configured -> emit a clean "skipped" result, exit 0 ----------
if [ -z "${USER_SIM_MODEL:-}" ]; then
  ae_log "tau2 SKIP: USER_SIM_MODEL is unset. tau2 needs a FIXED external user-simulator held"
  ae_log "constant across all 4 configs; using the model-under-test would confound the A/B."
  ae_log "Set USER_SIM_MODEL (+ USER_SIM_BASE_URL / USER_SIM_API_KEY) to run. See README.md."
  cat > "$PARSED" <<'JSON'
{"score": null, "score_name": "pass^1", "n_tasks": 0, "per_task": [],
 "extra": {"skipped": "no fixed user-sim configured (USER_SIM_MODEL unset)"}}
JSON
  NOW="$(ae_now)"
  emit "$PARSED" "NA NA" "NA NA" "$NOW" "$NOW"
  ae_log "tau2 skipped result written -> $OUT"
  exit 0
fi

# ---- subset -> (num_tasks, num_trials). Stable task ordering -> pairable task_ids --------------
DOMAIN="${TAU2_DOMAIN:-retail}"            # retail = 114 tasks (default A/B domain)
TRIALS=1                                    # greedy temp=0; pass^1 is the metric
case "$SUBSET" in
  smoke)    NUM_TASKS=3  ;;
  standard) NUM_TASKS=20 ;;
  full)     NUM_TASKS="" ;;                 # empty -> all tasks in the domain (retail 114)
  *) ae_log "tau2: unknown subset '$SUBSET' (smoke|standard|full)"; exit 2 ;;
esac

# Unique save dir under TAU2_DATA_DIR/simulations/<RUN_NAME>/results.json (tau2 hardcodes this base).
RUN_NAME="ae_${EVAL_LABEL}_${DOMAIN}_${SUBSET}"
NATIVE_DIR="$TAU2_DATA_DIR/simulations/$RUN_NAME"
RESULTS_JSON="$NATIVE_DIR/results.json"
rm -rf "$NATIVE_DIR"                         # fresh run (avoid tau2 auto-resume on stale partials)

# agent-llm-args: forwarded verbatim by tau2 -> litellm.completion (SOURCE-VERIFIED in
# src/tau2/utils/llm_utils.py: generate() splats **kwargs into completion()). api_base/api_key
# route the openai/<id> provider at our local vLLM endpoint. SMOKE-TEST this path first.
AGENT_ARGS=$("$AE_PY" - "$EVAL_ENDPOINT" "$AE_TEMPERATURE" "$AE_TOP_P" "$AE_MAX_TOKENS" "$AE_SEED" <<'PY'
import json, sys
base, temp, top_p, maxtok, seed = sys.argv[1:6]
print(json.dumps({
    "api_base": base, "api_key": "EMPTY",
    "temperature": float(temp), "top_p": float(top_p),
    "max_tokens": int(maxtok), "seed": int(seed),
}))
PY
)

# user-llm-args: temperature 0 (deterministic user sim too). Only set base_url/api_key if provided
# (a cloud user-sim typically reads its key from the provider's standard env var instead).
USER_ARGS=$("$AE_PY" - "${USER_SIM_BASE_URL:-}" "${USER_SIM_API_KEY:-}" <<'PY'
import json, sys
base, key = sys.argv[1], sys.argv[2]
d = {"temperature": 0.0}
if base: d["api_base"] = base
if key:  d["api_key"]  = key
print(json.dumps(d))
PY
)

ae_log "tau2 run: domain=$DOMAIN subset=$SUBSET tasks=${NUM_TASKS:-ALL} trials=$TRIALS"
ae_log "  agent-llm = openai/$EVAL_SERVED  @ $EVAL_ENDPOINT"
ae_log "  user-llm  = $USER_SIM_MODEL  (FIXED, held constant across configs)"
ae_log "  NOTE: vLLM /metrics token delta reflects the AGENT side only; the user-sim runs on a"
ae_log "        separate endpoint and is NOT counted here (correct -- we measure the model-under-test)."

# ---- run ---------------------------------------------------------------------------------------
B=$(ae_snap); START=$(ae_now)

CMD=( "$TAU2_BIN" run
  --domain "$DOMAIN"
  --agent-llm "openai/$EVAL_SERVED"
  --agent-llm-args "$AGENT_ARGS"
  --user-llm "$USER_SIM_MODEL"
  --user-llm-args "$USER_ARGS"
  --num-trials "$TRIALS"
  --max-concurrency "$AE_CONCURRENCY"
  --seed "$AE_SEED"
  --save-to "$RUN_NAME"
  --log-level WARNING )
[ -n "$NUM_TASKS" ] && CMD+=( --num-tasks "$NUM_TASKS" )

ae_log "+ ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RUNS_DIR/${RUN_NAME}.log"
RC=${PIPESTATUS[0]}

END=$(ae_now); A=$(ae_snap)

if [ "$RC" -ne 0 ] || [ ! -f "$RESULTS_JSON" ]; then
  ae_log "tau2 FAILED (rc=$RC) or no results.json at $RESULTS_JSON -- emitting null result"
  cat > "$PARSED" <<JSON
{"score": null, "score_name": "pass^1", "n_tasks": 0, "per_task": [],
 "extra": {"error": "tau2 run failed or produced no results.json (rc=$RC)"}}
JSON
  emit "$PARSED" "$B" "$A" "$START" "$END"
  exit "$RC"
fi

# Keep a copy of the native results under runs/ (gitignored) for debugging.
cp "$RESULTS_JSON" "$RUNS_DIR/${RUN_NAME}.results.json" 2>/dev/null || true

"$AE_PY" "$HERE/parse.py" "$RESULTS_JSON" > "$PARSED"
emit "$PARSED" "$B" "$A" "$START" "$END"
ae_log "tau2 done -> $OUT"
