#!/usr/bin/env bash
# agentic-eval/run/run_all.sh -- the campaign entry point. Serves each config in turn (SERIAL, both
# cards leased), runs the harness spectrum, tears the serve down (wedge-guarded), then summarizes.
#
# MUST be launched under the GPU lease so the whole campaign holds the box:
#     cd <repo> && ./bin/gpu-run bash agentic-eval/run/run_all.sh
#
# Knobs (env):
#     SUBSET=standard         smoke | standard | full
#     CONFIGS="27b-int4 ..."  subset/order of configs (default: all four)
#     HARNESSES="aider bfcl"  subset of harnesses (default: all wired)
#     KEEP_UP=0               1 = leave the LAST config serving (skip its teardown) for manual poking
#
# Serial-by-design: one config served at a time on PORT 18080 -> clean, uncontended wall-clock/token
# numbers (the timing metric is only meaningful if nothing else shares the cards). Parallel int4-on-
# both-cards is a future optimization (would contaminate timing), noted in README.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

SUBSET="${SUBSET:-standard}"
read -r -a CONFIGS <<< "${CONFIGS:-${EVAL_CONFIG_LIST[*]}}"
HARNESSES_ARGS=(); [ -n "${HARNESSES:-}" ] && read -r -a HARNESSES_ARGS <<< "$HARNESSES"

CUR_LABEL=""
cleanup() {  # never leave a serve (and the lease) dangling on crash/Ctrl-C
  [ -n "$CUR_LABEL" ] && { ae_log "trap: tearing down $CUR_LABEL"; bash "$AE_ROOT/serve/serve_config.sh" "$CUR_LABEL" stop || true; CUR_LABEL=""; }
}
trap cleanup EXIT INT TERM

ae_log "############ CAMPAIGN subset=$SUBSET configs=(${CONFIGS[*]}) ############"
CAMP_START=$(date +%s)
for label in "${CONFIGS[@]}"; do
  eval_config "$label" >/dev/null || { ae_log "skip unknown config $label"; continue; }
  ae_log "########## [$label] serve ##########"
  CUR_LABEL="$label"
  if ! bash "$AE_ROOT/serve/serve_config.sh" "$label" start; then
    ae_log "!!! [$label] serve FAILED -- skipping its harnesses"
    bash "$AE_ROOT/serve/serve_config.sh" "$label" stop || true
    CUR_LABEL=""; continue
  fi
  bash "$AE_ROOT/run/run_config.sh" "$label" "$SUBSET" "${HARNESSES_ARGS[@]}" || ae_log "[$label] run_config returned nonzero"
  if [ "${KEEP_UP:-0}" = 1 ] && [ "$label" = "${CONFIGS[-1]}" ]; then
    ae_log "[$label] KEEP_UP=1 -> leaving serve up"; CUR_LABEL=""
  else
    bash "$AE_ROOT/serve/serve_config.sh" "$label" stop || true; CUR_LABEL=""
  fi
done
trap - EXIT INT TERM

ae_log "########## summarize ##########"
"$AE_PY" "$AE_LIB/summarize.py" --results "$AE_ROOT/results" --readme "$AE_ROOT/README.md" || ae_log "summarize failed"
ae_log "############ CAMPAIGN done in $(( $(date +%s) - CAMP_START ))s ############"
