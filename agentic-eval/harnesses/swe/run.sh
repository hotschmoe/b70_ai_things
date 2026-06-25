#!/usr/bin/env bash
# agentic-eval/harnesses/swe/run.sh  --  bash run.sh <config_label> <subset>
#
# SWE harness: mini-swe-agent generates patches against the live vLLM endpoint
# (text/backticks bash mode -- no native tool-calls), then the official swebench
# grader scores them in per-instance docker containers on the HOST daemon.
# Emits the canonical results/<label>/swe.json via lib/evallib.py (HARNESS_CONTRACT.md).
#
# Subsets (STABLE dataset-order slices of SWE-bench Verified, so task_ids pair):
#   smoke    -> --slice 0:3     (plumbing shakeout, ~minutes once images cached)
#   standard -> --slice 0:20    (default campaign size)
#   full     -> --slice 0:100   (headline run)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$HERE/../.." && pwd)/lib/common.sh"
ae_set_config "${1:?usage: run.sh <config_label> <subset>}"
SUBSET="${2:-standard}"

VENV="$HERE/.venv"
PY="$VENV/bin/python"
MINI="$VENV/bin/mini-extra"
[ -x "$PY" ] || { ae_log "swe: venv missing -- run harnesses/swe/setup.sh first"; exit 1; }
[ -x "$MINI" ] || { ae_log "swe: mini-extra missing in venv -- run setup.sh"; exit 1; }

# ---- dataset + slice per subset ---------------------------------------------
# SWE-bench Verified (princeton-nlp/SWE-bench_Verified, split=test, 500 instances).
# We use Verified (curated, human-validated, the field-standard headline number)
# rather than rebench: it is the comparison everyone cites and its eval docker
# images are public+cached, so the 4-way quant delta is interpretable. The slice
# is taken in DATASET ORDER with NO --shuffle -> identical instance ids across all
# four configs -> lib/stats.py can pair them for McNemar/bootstrap.
SUBSET_NAME="verified"
SUBSET_SPLIT="test"
case "$SUBSET" in
  smoke)    SLICE="0:3" ;;
  standard) SLICE="0:20" ;;
  full)     SLICE="0:100" ;;
  *) ae_log "swe: unknown subset '$SUBSET' (smoke|standard|full)"; exit 2 ;;
esac

# ---- per-run output tree (gitignored: harnesses/*/runs/) --------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$HERE/runs/${EVAL_LABEL}/${SUBSET}-${STAMP}"
PREDS_DIR="$RUN_DIR/preds"        # mini writes preds.json + per-instance trajs here
REPORT_DIR="$RUN_DIR/report"      # swebench grader writes its report json here
mkdir -p "$PREDS_DIR" "$REPORT_DIR"
RUN_ID="ae-${EVAL_LABEL}-${SUBSET}-${STAMP}"

# ---- build the per-config model yaml from the template ----------------------
# Substitute the served id + endpoint + determinism knobs. Never hardcode a quant.
CFG="$RUN_DIR/config.yaml"
sed -e "s|@@SERVED@@|${EVAL_SERVED}|g" \
    -e "s|@@BASE_URL@@|${EVAL_BASE_URL}|g" \
    -e "s|@@TEMPERATURE@@|${AE_TEMPERATURE}|g" \
    -e "s|@@TOP_P@@|${AE_TOP_P}|g" \
    -e "s|@@MAX_TOKENS@@|${AE_MAX_TOKENS}|g" \
    "$HERE/config.yaml.tmpl" > "$CFG"
ae_log "swe: config -> $CFG (served=$EVAL_SERVED endpoint=$EVAL_BASE_URL temp=$AE_TEMPERATURE)"

export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # litellm/openai client wants a key present
export HF_HOME="${HF_HOME:-$HERE/data/hf}"         # reuse the dataset cache from setup.sh
# Keep mini's own cost guard from ever aborting a local run:
export MSWEA_COST_TRACKING=ignore_errors

ae_log "swe: === $EVAL_LABEL :: $SUBSET :: $SUBSET_NAME[$SLICE] ==="

# ---- token accounting + wall-clock snapshot (HARNESS_CONTRACT.md) -----------
B=$(ae_snap); START=$(ae_now)

# 1) generate predictions with mini-swe-agent (bash/backticks mode).
#    -c swebench_backticks.yaml  -> builtin text/bash prompt (no tool-calls)
#    -c $CFG                     -> our local-vllm model wiring (merged on top)
#    --environment-class docker  -> per-instance SWE-bench eval image on the host daemon
ae_log "swe: mini-extra swebench (workers=$AE_CONCURRENCY) ..."
"$MINI" swebench \
  --subset "$SUBSET_NAME" --split "$SUBSET_SPLIT" --slice "$SLICE" \
  --workers "$AE_CONCURRENCY" \
  --environment-class docker \
  -c swebench_backticks.yaml -c "$CFG" \
  -o "$PREDS_DIR"
MINI_RC=$?
ae_log "swe: mini-extra rc=$MINI_RC"

PREDS="$PREDS_DIR/preds.json"
if [ ! -s "$PREDS" ]; then
  ae_log "swe: NO predictions produced ($PREDS missing/empty) -- aborting grade"
  END=$(ae_now); A=$(ae_snap)
  # still emit an (empty) result so the campaign records the failure rather than a gap.
  printf '{"score":0.0,"score_name":"resolved","n_tasks":0,"per_task":[],"extra":{"error":"no_predictions"}}\n' \
    > "$RESULTS_DIR/.swe.parsed.json"
else
  # 2) grade with the official swebench harness (per-instance docker, host daemon).
  ae_log "swe: grading with swebench harness (run_id=$RUN_ID, max_workers=$AE_CONCURRENCY) ..."
  ( cd "$REPORT_DIR" && "$PY" -m swebench.harness.run_evaluation \
      --dataset_name "princeton-nlp/SWE-bench_Verified" \
      --split "$SUBSET_SPLIT" \
      --predictions_path "$PREDS" \
      --run_id "$RUN_ID" \
      --max_workers "$AE_CONCURRENCY" \
      --report_dir "$REPORT_DIR" \
      --cache_level env )
  GRADE_RC=$?
  ae_log "swe: grader rc=$GRADE_RC"

  END=$(ae_now); A=$(ae_snap)

  # 3) parse the grader report + preds -> canonical parsed.json
  "$PY" "$HERE/parse.py" "$RUN_DIR" > "$RESULTS_DIR/.swe.parsed.json" || {
    ae_log "swe: parse.py failed"; exit 1; }
fi
# (END/A already set in the no-preds branch above; ensure set for the happy path)
END="${END:-$(ae_now)}"; A="${A:-$(ae_snap)}"

# 4) emit the standard per-(config,harness) record.
"$AE_PY" "$AE_LIB/evallib.py" emit \
  --config "$EVAL_LABEL" --harness swe --subset "$SUBSET" --served "$EVAL_SERVED" \
  --parsed "$RESULTS_DIR/.swe.parsed.json" \
  --tok-before "$B" --tok-after "$A" --start "$START" --end "$END" \
  --out "$RESULTS_DIR/swe.json" --meta "temperature=$AE_TEMPERATURE" \
  --meta "subset_dataset=$SUBSET_NAME" --meta "slice=$SLICE" --meta "run_id=$RUN_ID"

ae_log "swe: done -> $RESULTS_DIR/swe.json (raw under $RUN_DIR)"
