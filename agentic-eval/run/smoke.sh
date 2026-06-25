#!/usr/bin/env bash
# agentic-eval/run/smoke.sh [label] [harness ...] -- plumbing shakeout on ONE config.
# Serves the config, runs the harness spectrum at subset=smoke (tiny task counts), tears down,
# summarizes. This is what you run FIRST to de-risk endpoint/handler/grader wiring before a campaign.
#
#     cd <repo> && ./bin/gpu-run bash agentic-eval/run/smoke.sh 27b-w8a8
#
# Default config is 27b-w8a8 (the current shelf default). Pass a label + optional harness subset.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

LABEL="${1:-27b-w8a8}"; shift || true
eval_config "$LABEL" >/dev/null || { echo "unknown config $LABEL (valid: ${EVAL_CONFIG_LIST[*]})"; exit 2; }

CUR="$LABEL"
trap '[ -n "$CUR" ] && bash "$AE_ROOT/serve/serve_config.sh" "$CUR" stop || true' EXIT INT TERM
ae_log "SMOKE: serve $LABEL"
bash "$AE_ROOT/serve/serve_config.sh" "$LABEL" start || { ae_log "serve failed"; exit 1; }
ae_set_config "$LABEL"
ae_log "SMOKE: endpoint self-test (identity / metrics / completion / tool-call)"
if ae_endpoint_selftest; then
  bash "$AE_ROOT/run/run_config.sh" "$LABEL" smoke "$@" || true
else
  ae_log "SMOKE: endpoint self-test FAILED (no generation) -- skipping harnesses, tearing down"
fi
bash "$AE_ROOT/serve/serve_config.sh" "$LABEL" stop || true
CUR=""; trap - EXIT INT TERM
"$AE_PY" "$AE_LIB/summarize.py" --results "$AE_ROOT/results" || true
ae_log "SMOKE done for $LABEL"
