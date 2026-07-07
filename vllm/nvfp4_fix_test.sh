#!/usr/bin/env bash
# NVFP4 TP=1 single-card fix A/B for the MTP+capture NEO abort (fast repro ~1-2k tok).
# serve (MODE=fused GRAPH=1 MTPTOK=5 KV_FP8=0 + fix env) -> forced-decode soak -> crash/survival
# + container log -> teardown. Set the fix via B70_EXTRA_ENV before calling.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things; cd "$REPO"
CFG_LABEL="${CFG_LABEL:-nvfp4}"
NAME="${NAME:-nvfp4_repro}"; PORT="${PORT:-8078}"; CARD="${CARD:-0}"; TP="${TP:-2}"
LOGDIR="$REPO/results/logs"; mkdir -p "$LOGDIR"; RESULTS="$LOGDIR/neo_abort_campaign.txt"
STAMP="$(date +%Y%m%d_%H%M%S)"
SLOG="$LOGDIR/nvfp4_${CFG_LABEL}_${STAMP}.serve.log"; KLOG="$LOGDIR/nvfp4_${CFG_LABEL}_${STAMP}.soak.log"
CLOG="$LOGDIR/nvfp4_${CFG_LABEL}_${STAMP}.container.log"
export WORKERS="${WORKERS:-1}" CTX_CHARS="${CTX_CHARS:-6000}" MAXTOK="${MAXTOK:-3000}"
export CEIL_TOK="${CEIL_TOK:-6000}" CEIL_SEC="${CEIL_SEC:-900}" PROBE_HOST="${PROBE_HOST:-localhost}"

echo "===== [NVFP4 $CFG_LABEL] B70_EXTRA_ENV='${B70_EXTRA_ENV:-}' TP=$TP card=$CARD soak_ceil=$CEIL_TOK/$CEIL_SEC $(date -u +%FT%TZ) =====" | tee -a "$RESULTS"
docker rm -f "$NAME" >/dev/null 2>&1

# Launch NVFP4 crashing config (bf16 KV to isolate from the repetition fault). TP=2 (default) records
# the capture-safe all-reduce into the graph per replay -> the fast crash the DD hit; TP=1 has no collectives.
_MAXLEN=$([ "$TP" = 2 ] && echo 131072 || echo 8192)
NAME="$NAME" PORT="$PORT" CARD="$CARD" TP="$TP" MODE=fused GRAPH=1 MTPTOK=5 KV_FP8=0 \
  CAPSIZES=1,2,4,8 MAXLEN="${MAXLEN:-$_MAXLEN}" UTIL=0.85 B70_DEBUG=1 \
  bash vllm/nvfp4/serve_nvfp4_27b.sh start > "$SLOG" 2>&1
echo "[NVFP4 $CFG_LABEL] container launched; waiting for /health..." | tee -a "$RESULTS"

# poll health (capture ~2-3min single card)
ok=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  if ! docker ps --filter name="$NAME" --format '{{.Names}}' | grep -q "$NAME"; then echo "[NVFP4 $CFG_LABEL] container died during startup" | tee -a "$RESULTS"; break; fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  echo "[NVFP4 $CFG_LABEL] NEVER HEALTHY -> abort" | tee -a "$RESULTS"; docker logs "$NAME" > "$CLOG" 2>&1
  grep -E 'drafter-eager|cg-recycle|Error|Traceback' "$CLOG" | tail -8 | tee -a "$RESULTS"
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
echo "[NVFP4 $CFG_LABEL] HEALTHY" | tee -a "$RESULTS"
docker logs "$NAME" 2>&1 | grep -iE 'drafter-eager|cg-recycle|nvfp4-shim.*register_fake|FIX-SYNC' | tail -4 | tee -a "$RESULTS"

PORT="$PORT" KEY="" python3 vllm/soak_concurrent.py "$CFG_LABEL" > "$KLOG" 2>&1
SOAK_RC=$?
tail -3 "$KLOG" | tee -a "$RESULTS"
echo "[NVFP4 $CFG_LABEL] soak rc=$SOAK_RC (3=crash)" | tee -a "$RESULTS"
docker logs "$NAME" > "$CLOG" 2>&1
echo "abort-lines=$(grep -cE 'Abort was called|linear_stream' "$CLOG")" | tee -a "$RESULTS"
grep -E 'Abort was called|linear_stream|in replay|in propose|drafter-eager|Fatal Python' "$CLOG" | tail -8 | tee -a "$RESULTS"

docker rm -f "$NAME" >/dev/null 2>&1
echo "[NVFP4 $CFG_LABEL] done $(date -u +%FT%TZ)" | tee -a "$RESULTS"; echo | tee -a "$RESULTS"
exit "$SOAK_RC"
