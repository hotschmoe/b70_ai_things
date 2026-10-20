#!/usr/bin/env bash
# Ornith T2: health, one-card serve, real second turn, health.
#
# serve_ornith.sh is the gpu-run holder. This script does not wrap that
# serve in a second gpu-run. The client is a second container with no
# device. Copy path is still swap_blocks_batch.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CARD="${CARD:-0}"
PORT="${PORT:-18111}"
NAME="vllm-fn-g4-kv-c${CARD}"
LOGDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/logs
TAG="${TAG:-}"
SFX=""
if [[ -n "$TAG" ]]; then
  SFX="-${TAG}"
fi
SERVE_LOG="$LOGDIR/g4-t2-serve-c${CARD}${SFX}.txt"
OUT="$LOGDIR/g4-t2-turn-c${CARD}${SFX}.json"
FORK=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm
SITE=/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages
MODEL="$REPO/models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2"
IMG=vllm-flashnext-xpu:py312
READY_TIMEOUT="${READY_TIMEOUT:-1500}"

export B70_AGENT=g4
mkdir -p "$LOGDIR"
rm -f "$OUT"

echo "T2 health before $(date '+%F %T') card=$CARD tag=${TAG:-swap} copy=${B70_KVPARK_COPY:-0}"
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-t2-health-before-c${CARD}${SFX}.txt" 2>&1
echo "T2 health before ok"

echo "T2 serve start $(date '+%F %T')"
: > "$SERVE_LOG"
# xpu-run waits forever when this is unset. A lost race should fail.
B70_GPU_LOCK_TIMEOUT=180 CARD="$CARD" PORT="$PORT" MEMORY=80g \
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
    echo "T2 SERVE_DIED before ready"
    tail -40 "$SERVE_LOG"
    serve_rc=1
    break
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" | grep -q hotschmoe-dev; then
    if grep -q "G4 parking:" "$SERVE_LOG"; then
      echo "T2 READY $(date '+%F %T')"
      break
    fi
  fi
  sleep 5
done
if [[ "$serve_rc" == 0 ]] && ! curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" | grep -q hotschmoe-dev; then
  echo "T2 NOT_READY"
  tail -40 "$SERVE_LOG"
  serve_rc=1
fi
if [[ "$serve_rc" == 0 ]] && ! grep -q "G4 parking:" "$SERVE_LOG"; then
  echo "T2 connector did not log parking"
  grep -E "G4 |Error|Traceback" "$SERVE_LOG" | tail -30
  serve_rc=1
fi

rc=$serve_rc
if [[ "$serve_rc" == 0 ]]; then
echo "T2 client $(date '+%F %T')"
set +e
CLIENT_MAX_LEN="${CLIENT_MAX_LEN:-32768}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
P1_ARGS=()
if [[ -n "${P1_TOKENS:-}" ]]; then
  P1_ARGS+=(--p1-tokens "$P1_TOKENS")
fi
docker run --rm --network host --cpus 2 --memory 8g \
  -v /mnt/vm_8tb:/mnt/vm_8tb \
  -e PYTHONPATH="${FORK}:${REPO}/vllm/flashnext:${SITE}" \
  -e VLLM_TARGET_DEVICE=cpu \
  -e ONEAPI_DEVICE_SELECTOR=opencl:cpu \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /usr/bin/python3.12 \
  "$IMG" \
  "$REPO/vllm/flashnext/kvoffload/turn_test.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --serve-log "$SERVE_LOG" \
  --tokenizer "$MODEL" \
  --out "$OUT" \
  --max-model-len "$CLIENT_MAX_LEN" \
  --request-timeout "$REQUEST_TIMEOUT" \
  "${P1_ARGS[@]}"
rc=$?
set -e
echo "T2 client rc=$rc $(date '+%F %T')"
if [[ -f "$OUT" ]]; then
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("T2_RESULT", "pass="+str(r.get("pass")), "host="+str(r.get("host_delta")), "device="+str(r.get("device_delta")), "computed_ok="+str(r.get("computed_ok")), "greedy="+str(r.get("greedy_match")))' "$OUT"
fi
fi

echo "T2 stop serve $(date '+%F %T')"
docker stop -t 25 "$NAME" >/dev/null 2>&1 || true
wait "$SPID" 2>/dev/null || true
trap - EXIT

echo "T2 health after $(date '+%F %T')"
# A lost lock must not replace a passing client exit. The printed
# rc is the health command. The script exit stays the client's.
set +e
B70_GPU_LOCK_TIMEOUT=90 "$REPO/bin/gpu-run" --card "$CARD" \
  "$REPO/bin/xpu-health" --card "$CARD" \
  > "$LOGDIR/g4-t2-health-after-c${CARD}${SFX}.txt" 2>&1
health_rc=$?
set -e
echo "T2 health after rc=$health_rc"
tail -5 "$LOGDIR/g4-t2-health-after-c${CARD}${SFX}.txt"
exit "$rc"
