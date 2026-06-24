#!/usr/bin/env bash
# agentic-eval/run/run_config.sh <label> [subset] [harness ...]
#
# Runs the harness spectrum against an ALREADY-SERVING endpoint for <label> (run_all.sh handles the
# serve lifecycle + lease). Each harness writes results/<label>/<harness>.json via the standard emit.
#
#   subset    smoke | standard | full   (default standard; passed through to each harness run.sh)
#   harness   subset of: aider bfcl tau2 swe   (default: all that have a run.sh)
#
# Harnesses are independent; one failing does not abort the rest (its result is just absent/null).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

LABEL="${1:?usage: run_config.sh <label> [subset] [harness ...]}"; shift || true
SUBSET="${1:-standard}"; shift || true
ae_set_config "$LABEL" || exit 2

# Default harness set = every harnesses/*/run.sh that exists, in spectrum order.
DEFAULT_ORDER=(aider bfcl tau2 swe)
HARNESSES=("$@")
if [ "${#HARNESSES[@]}" -eq 0 ]; then
  for h in "${DEFAULT_ORDER[@]}"; do
    [ -x "$AE_ROOT/harnesses/$h/run.sh" ] || [ -f "$AE_ROOT/harnesses/$h/run.sh" ] && HARNESSES+=("$h")
  done
fi

ae_wait_endpoint "$EVAL_SERVED" 180 || { ae_log "no endpoint for $LABEL -- skipping harnesses"; exit 1; }

ae_log "=== config $LABEL :: subset=$SUBSET :: harnesses: ${HARNESSES[*]:-none} ==="
for h in "${HARNESSES[@]}"; do
  rsh="$AE_ROOT/harnesses/$h/run.sh"
  [ -f "$rsh" ] || { ae_log "harness '$h' has no run.sh -- skip"; continue; }
  ae_log "--- harness $h ($LABEL) start ---"
  if bash "$rsh" "$LABEL" "$SUBSET"; then
    ae_log "--- harness $h ($LABEL) done ---"
  else
    ae_log "!!! harness $h ($LABEL) FAILED (rc=$?) -- continuing"
  fi
done
ae_log "=== config $LABEL complete ==="
