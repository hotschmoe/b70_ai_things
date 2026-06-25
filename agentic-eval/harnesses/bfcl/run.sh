#!/usr/bin/env bash
# agentic-eval/harnesses/bfcl/run.sh -- BFCL (Berkeley Function Calling Leaderboard) multi_turn harness.
#
#   bash run.sh <config_label> <subset>
#     subset = smoke | standard | full
#
# Drives BFCL v4 multi_turn against our ALREADY-RUNNING vLLM OpenAI-compatible endpoint
# (--skip-server-setup), using a runtime-registered model entry whose key == $EVAL_SERVED so the
# OpenAI `model` field matches what vLLM serves. See README.md for the MODEL_KEY decision + why.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$HERE"/../.. && pwd)/lib/common.sh"
ae_set_config "$1"
SUBSET="${2:-standard}"

PY="$HERE/.venv/bin/python"
BFCL="$HERE/.venv/bin/bfcl"
[ -x "$PY" ]   || { echo "FATAL: .venv missing -- run setup.sh first"; exit 1; }
[ -x "$BFCL" ] || { echo "FATAL: bfcl CLI missing -- run setup.sh first"; exit 1; }

# ---- BFCL workspace (result/ + score/ + .env + id-file all live here, not in site-packages) -----
WORK="$HERE/work/$EVAL_LABEL"
mkdir -p "$WORK"
export BFCL_PROJECT_ROOT="$WORK"

# ---- point BFCL's stock Qwen OSS handler at our running endpoint --------------------------------
# base_oss_handler.py reads these. REMOTE_OPENAI_BASE_URL must be the .../v1 root.
export REMOTE_OPENAI_BASE_URL="$EVAL_ENDPOINT"
export REMOTE_OPENAI_API_KEY="dummy"

# The handler tokenizes the prompt locally only to CAP max_tokens (context math); it never downloads
# if we hand it a local tokenizer dir. Map served-id -> the on-disk model dir for this config.
case "$EVAL_SERVED" in
  qwen36-27b-int4)                 TOK=/mnt/vm_8tb/b70/models/Lorbus_Qwen3.6-27B-int4-AutoRound ;;
  qwen36-27b-w8a8-sqgptq-mtp)      TOK=/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft ;;
  qwen36-35b-a3b-int4)             TOK=/mnt/vm_8tb/b70/models/Intel_Qwen3.6-35B-A3B-int4-AutoRound ;;
  qwen36-35b-a3b-quark-w8a8-int8)  TOK=/mnt/vm_8tb/b70/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8 ;;
  *)                               TOK="" ;;
esac
if [ -n "$TOK" ] && [ -f "$TOK/tokenizer_config.json" ]; then
  export REMOTE_OPENAI_TOKENIZER_PATH="$TOK"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  ae_log "WARN: no local tokenizer for $EVAL_SERVED; BFCL will try to fetch ${EVAL_SERVED} from HF"
fi

# ---- register the served-id as a BFCL model key (sitecustomize.py runs in the bfcl subprocess) --
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export BFCL_REGISTER_MODEL="$EVAL_SERVED"
export BFCL_REGISTER_FC="1"            # FC = prompt-injected tools + <tool_call> XML parse (Qwen3 native format)
MODEL_KEY="$EVAL_SERVED"

# ---- subset -> test selection -------------------------------------------------------------------
# standard/full = the whole multi_turn collection (4 subcats x 200 = 800 cases).
# smoke = a tiny fixed id slice via BFCL's --run-ids + test_case_ids_to_generate.json (5 cases).
RUN_IDS_ARGS=()
EVAL_PARTIAL_ARGS=()
case "$SUBSET" in
  smoke)
    cat > "$WORK/test_case_ids_to_generate.json" <<'JSON'
{
  "multi_turn_base": ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_base_2"],
  "multi_turn_miss_func": ["multi_turn_miss_func_0"],
  "multi_turn_miss_param": ["multi_turn_miss_param_0"]
}
JSON
    RUN_IDS_ARGS=(--run-ids)
    # Evaluating only a subset of a category requires --partial-eval, else bfcl raises on the
    # result/dataset length mismatch.
    EVAL_PARTIAL_ARGS=(--partial-eval)
    ;;
  standard|full)
    : ;;  # full multi_turn (all 800)
  *)
    echo "FATAL: unknown subset '$SUBSET' (smoke|standard|full)"; exit 2 ;;
esac

ae_log "bfcl multi_turn: config=$EVAL_LABEL served=$EVAL_SERVED subset=$SUBSET endpoint=$EVAL_ENDPOINT"

# ---- token/wall accounting wrapper (contract) ---------------------------------------------------
B=$(ae_snap); START=$(ae_now)

# generate: hit the live endpoint. --temperature from determinism knobs; --num-threads = concurrency.
# -o allows overwriting a prior run's results so re-runs are clean.
"$BFCL" generate \
  --model "$MODEL_KEY" \
  --test-category multi_turn \
  --skip-server-setup \
  --num-threads "$AE_CONCURRENCY" \
  --temperature "$AE_TEMPERATURE" \
  -o \
  "${RUN_IDS_ARGS[@]}"
GEN_RC=$?

# evaluate: score the generated results into $WORK/score/...
"$BFCL" evaluate \
  --model "$MODEL_KEY" \
  --test-category multi_turn \
  "${EVAL_PARTIAL_ARGS[@]}"
EVAL_RC=$?

END=$(ae_now); A=$(ae_snap)

if [ $GEN_RC -ne 0 ]; then ae_log "WARN: bfcl generate exited $GEN_RC"; fi
if [ $EVAL_RC -ne 0 ]; then ae_log "WARN: bfcl evaluate exited $EVAL_RC"; fi

# ---- parse native score dir -> parsed.json -> canonical emit ------------------------------------
MODEL_DIR="${MODEL_KEY//\//_}"          # BFCL escapes '/' to '_' in dir names
SCORE_DIR="$WORK/score/$MODEL_DIR/multi_turn"
RESULT_DIR_NATIVE="$WORK/result/$MODEL_DIR/multi_turn"

"$PY" "$HERE/parse.py" "$SCORE_DIR" --result-dir "$RESULT_DIR_NATIVE" \
  > "$RESULTS_DIR/.bfcl.parsed.json"

"$AE_PY" "$AE_LIB/evallib.py" emit \
  --config "$EVAL_LABEL" --harness bfcl --subset "$SUBSET" --served "$EVAL_SERVED" \
  --parsed "$RESULTS_DIR/.bfcl.parsed.json" \
  --tok-before "$B" --tok-after "$A" --start "$START" --end "$END" \
  --out "$RESULTS_DIR/bfcl.json" --meta "temperature=$AE_TEMPERATURE" --meta "model_key=$MODEL_KEY"
