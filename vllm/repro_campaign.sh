#!/usr/bin/env bash
# One repro config end-to-end under the GPU lease:
#   serve.sh start -> forced-decode probe -> capture container log -> serve.sh stop.
# Env knobs (all optional): CFG_LABEL, MAXLEN, extra serve env via SERVE_ENV="A=1 B=2",
# probe knobs PROBE_CTX_CHARS/PROBE_MAXTOK/PROBE_RUNS. Writes a result line to RESULTS.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
cd "$REPO"
CFG_LABEL="${CFG_LABEL:-baseline}"
NAME="${NAME:-b70_repro}"
PORT="${PORT:-18091}"
MAXLEN="${MAXLEN:-131072}"
KEYFILE="${KEYFILE:-/tmp/repro_ddkey}"
RESULTS="${RESULTS:-$REPO/results/logs/neo_abort_campaign.txt}"
LOGDIR="$REPO/results/logs"
mkdir -p "$LOGDIR"
[ -f "$KEYFILE" ] || echo "reprokey-$(date +%s)" > "$KEYFILE"
KEY="$(cat "$KEYFILE")"
STAMP="$(date +%Y%m%d_%H%M%S)"
SLOG="$LOGDIR/repro_${CFG_LABEL}_${STAMP}.serve.log"
PLOG="$LOGDIR/repro_${CFG_LABEL}_${STAMP}.probe.log"

echo "===== [$CFG_LABEL] MAXLEN=$MAXLEN SERVE_ENV='${SERVE_ENV:-}' $(date -u +%FT%TZ) =====" | tee -a "$RESULTS"

# Launch serve (detached container inside). serve.sh reads B70_DEBUG, B70_NOMTP,
# B70_EXTRA_ENV, NOMM, GRAPH, MTPTOK, CAPSIZES etc. directly from the inherited env
# (set them on the campaign's command line). Only NAME/PORT/TP/MAXLEN/API_KEY are
# forced here so a multi-KV B70_EXTRA_ENV="A=1 B=2" survives intact.
NAME="$NAME" PORT="$PORT" TP=2 MAXLEN="$MAXLEN" API_KEY="$KEY" \
    bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start > "$SLOG" 2>&1
START_RC=$?
echo "[$CFG_LABEL] serve start rc=$START_RC (log: $SLOG)" | tee -a "$RESULTS"

if [ "$START_RC" -ne 0 ]; then
  echo "[$CFG_LABEL] SERVE FAILED TO START -> abort config" | tee -a "$RESULTS"
  tail -30 "$SLOG" | tee -a "$RESULTS"
  bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh stop >/dev/null 2>&1
  exit 1
fi

# Drive the forced-decode probe.
PORT="$PORT" KEY="$KEY" python3 vllm/repro_neo_abort.py "$CFG_LABEL" > "$PLOG" 2>&1
PROBE_RC=$?
echo "[$CFG_LABEL] probe rc=$PROBE_RC (3=crash observed)" | tee -a "$RESULTS"
grep -E 'VERDICTS|run[0-9]:' "$PLOG" | tee -a "$RESULTS"

# Capture any NEO abort / faulthandler traceback from the container before teardown.
echo "--- container log tail (abort search) ---" | tee -a "$RESULTS"
docker logs "$NAME" 2>&1 | grep -nE 'Abort was called|linear_stream|faulthandler|Fatal Python|in replay|in forward|in propose|EngineDeadError|cancelled|LinearStream|command buffer' | tail -25 | tee -a "$RESULTS"

NAME="$NAME" bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh stop > "$LOGDIR/repro_${CFG_LABEL}_${STAMP}.stop.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1   # belt-and-suspenders: ensure the repro container is gone
echo "[$CFG_LABEL] stopped. $(date -u +%FT%TZ)" | tee -a "$RESULTS"
echo | tee -a "$RESULTS"
exit "$PROBE_RC"
