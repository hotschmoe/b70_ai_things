#!/usr/bin/env bash
# Fix A/B: serve DD config (+ whatever B70_* fix env is set) -> concurrent soak past the
# baseline crash threshold (~96k tok) -> report survival/crash + abort trace -> teardown.
# Set the fix via env before calling, e.g. B70_EXTRA_ENV="B70_XPU_CG_SYNC_STEPS=1".
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things; cd "$REPO"
CFG_LABEL="${CFG_LABEL:-fix}"
NAME="${NAME:-b70_repro}"; PORT="${PORT:-18091}"; MAXLEN="${MAXLEN:-131072}"
KEYFILE="${KEYFILE:-/tmp/repro_ddkey}"; [ -f "$KEYFILE" ] || echo "reprokey-$(date +%s)" > "$KEYFILE"
KEY="$(cat "$KEYFILE")"
LOGDIR="$REPO/results/logs"; mkdir -p "$LOGDIR"
RESULTS="$LOGDIR/neo_abort_campaign.txt"
STAMP="$(date +%Y%m%d_%H%M%S)"
SLOG="$LOGDIR/fix_${CFG_LABEL}_${STAMP}.serve.log"; KLOG="$LOGDIR/fix_${CFG_LABEL}_${STAMP}.soak.log"
# soak params (default: push to ~2.3x the ~96k baseline threshold, or 45min)
export WORKERS="${WORKERS:-6}" CTX_CHARS="${CTX_CHARS:-28000}" MAXTOK="${MAXTOK:-4000}"
export CEIL_TOK="${CEIL_TOK:-220000}" CEIL_SEC="${CEIL_SEC:-3300}"

echo "===== [FIX $CFG_LABEL] B70_EXTRA_ENV='${B70_EXTRA_ENV:-}' MAXLEN=$MAXLEN soak_ceil=$CEIL_TOK/$CEIL_SEC $(date -u +%FT%TZ) =====" | tee -a "$RESULTS"
docker rm -f "$NAME" >/dev/null 2>&1

NAME="$NAME" PORT="$PORT" TP=2 MAXLEN="$MAXLEN" API_KEY="$KEY" \
  bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start > "$SLOG" 2>&1
if [ $? -ne 0 ]; then echo "[FIX $CFG_LABEL] SERVE FAILED" | tee -a "$RESULTS"; tail -25 "$SLOG" | tee -a "$RESULTS"; NAME="$NAME" bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh stop >/dev/null 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; fi
# confirm the fix block actually loaded
grep -E 'cg-sync|cg-recycle|FIX-SYNC|Tier F ENABLED' "$SLOG" | tail -3 | tee -a "$RESULTS"

PORT="$PORT" KEY="$KEY" python3 vllm/soak_concurrent.py "$CFG_LABEL" > "$KLOG" 2>&1
SOAK_RC=$?
tail -3 "$KLOG" | tee -a "$RESULTS"
echo "[FIX $CFG_LABEL] soak rc=$SOAK_RC (3=crash)" | tee -a "$RESULTS"
echo "--- abort trace? ---" | tee -a "$RESULTS"
docker logs "$NAME" 2>&1 | grep -cE 'Abort was called|linear_stream' | xargs -I{} echo "abort-lines={}" | tee -a "$RESULTS"

NAME="$NAME" bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh stop > "$LOGDIR/fix_${CFG_LABEL}_${STAMP}.stop.log" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "[FIX $CFG_LABEL] done $(date -u +%FT%TZ)" | tee -a "$RESULTS"; echo | tee -a "$RESULTS"
exit "$SOAK_RC"
