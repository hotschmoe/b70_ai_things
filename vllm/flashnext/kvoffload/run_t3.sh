#!/usr/bin/env bash
# Ornith T3. serve_ornith.sh is the only gpu-run holder.
# PHASE=itl is the two-decode stall check. PHASE=cycle is 4 parked
# plus 2 active. PHASE=both runs them on one serve.
# Does not enable the direct copy unless B70_KVPARK_COPY is already 1.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CARD="${CARD:-0}"
PORT="${PORT:-18112}"
PHASE="${PHASE:-itl}"
NAME="vllm-fn-g4-kv-c${CARD}"
LOGDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/logs
SERVE_LOG="$LOGDIR/g4-t3-serve-c${CARD}.txt"
OUT="$LOGDIR/g4-t3-${PHASE}-c${CARD}.json"
FORK=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm
SITE=/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages
MODEL="$REPO/models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2"
IMG=vllm-flashnext-xpu:py312
READY_TIMEOUT="${READY_TIMEOUT:-1500}"

export B70_AGENT=g4
mkdir -p "$LOGDIR"
rm -f "$OUT"

echo "T3 health before $(date '+%F %T') card=$CARD phase=$PHASE copy=${B70_KVPARK_COPY:-0}"
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-t3-health-before-c${CARD}.txt" 2>&1
echo "T3 health before ok"

echo "T3 serve start $(date '+%F %T')"
: > "$SERVE_LOG"
# itl keeps the T2 cap (45) so two short decodes fit beside the restore.
# cycle uses 32 blocks. The client keeps both actives on short prompts
# so a parked restore can allocate while they decode.
# Both phases on one serve stay at 45.
GPU_BLOCKS=45
if [[ "$PHASE" == "cycle" ]]; then
  GPU_BLOCKS=32
  export MAX_LEN="${MAX_LEN:-27000}"
fi
B70_GPU_LOCK_TIMEOUT=180 CARD="$CARD" PORT="$PORT" MEMORY=80g \
  MAX_NUM_SEQS=4 NUM_GPU_BLOCKS="$GPU_BLOCKS" \
  "$HERE/serve_ornith.sh" >> "$SERVE_LOG" 2>&1 &
SPID=$!

cleanup() {
  docker stop -t 25 "$NAME" >/dev/null 2>&1 || true
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + READY_TIMEOUT))
serve_rc=0
while (( SECONDS < deadline )); do
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "T3 SERVE_DIED before ready"
    tail -40 "$SERVE_LOG"
    serve_rc=1
    break
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" | grep -q hotschmoe-dev; then
    if grep -q "G4 parking:" "$SERVE_LOG"; then
      echo "T3 READY $(date '+%F %T')"
      break
    fi
  fi
  sleep 5
done
if [[ "$serve_rc" == 0 ]] && ! grep -q "G4 parking:" "$SERVE_LOG"; then
  echo "T3 NOT_READY"
  tail -30 "$SERVE_LOG"
  serve_rc=1
fi

rc=$serve_rc
if [[ "$serve_rc" == 0 ]]; then
  echo "T3 client $(date '+%F %T') phase=$PHASE"
  # Quiet CPU only for the timed client. The serve already holds gpu-run.
  set +e
  B70_CPU_LOCK_TIMEOUT=180 "$REPO/bin/cpu-run" --quiet \
  docker run --rm --network host --cpus 2 --memory 8g \
    -v /mnt/vm_8tb:/mnt/vm_8tb \
    -e PYTHONPATH="${FORK}:${REPO}/vllm/flashnext:${SITE}" \
    -e VLLM_TARGET_DEVICE=cpu \
    -e ONEAPI_DEVICE_SELECTOR=opencl:cpu \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /usr/bin/python3.12 \
    "$IMG" \
    "$REPO/vllm/flashnext/kvoffload/t3_test.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --serve-log "$SERVE_LOG" \
    --tokenizer "$MODEL" \
    --out "$OUT" \
    --phase "$PHASE" \
    --max-model-len 32768
  rc=$?
  set -e
  echo "T3 client rc=$rc $(date '+%F %T')"
  if [[ -f "$OUT" ]]; then
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("T3_RESULT", "pass="+str(r.get("pass")), "phase="+str(r.get("phase")))' "$OUT"
  fi
fi

echo "T3 stop serve $(date '+%F %T')"
docker stop -t 25 "$NAME" >/dev/null 2>&1 || true
wait "$SPID" 2>/dev/null || true
trap - EXIT

echo "T3 health after $(date '+%F %T')"
# A lost lock must not replace a passing client exit.
set +e
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-t3-health-after-c${CARD}.txt" 2>&1
health_rc=$?
set -e
echo "T3 health after rc=$health_rc"
tail -5 "$LOGDIR/g4-t3-health-after-c${CARD}.txt"
exit "$rc"
