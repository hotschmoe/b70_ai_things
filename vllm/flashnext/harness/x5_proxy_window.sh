#!/usr/bin/env bash
# One bounded serve per lease; releases the card between comparison arms.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
export PYTHONIOENCODING=ascii:backslashreplace
export QUANT=${QUANT:-UD-IQ3_XXS} MTP_TOKENS=${MTP_TOKENS:-0} CARD=${CARD:-1} PORT=${PORT:-18125}
export NAME=vllm-fn-x5-${QUANT,,}-mtp${MTP_TOKENS}
TAG=${TAG:-${QUANT,,}-mtp${MTP_TOKENS}-gdn${B70_X5_GDN_BF16:-0}-ev${B70_X5_EVENTLESS:-0}}
OUT=$B/x5/$TAG
mkdir -p "$OUT"
test ! -e "$OUT/started" || { echo "Refusing to overwrite $OUT"; exit 2; }
touch "$OUT/started"
exec > >(tee -a "$OUT/window.log") 2>&1
date -u
systemctl --user is-active hotschmoe-dd > "$OUT/dd-state.txt" || true
if grep -qx active "$OUT/dd-state.txt"; then echo DD_ACTIVE_ABORT; exit 3; fi
"$REPO/bin/gpu-run" --card "$CARD" "$REPO/bin/xpu-health" --card "$CARD" > "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
python3 "$REPO/vllm/flashnext/harness/x5_manifest.py" "$OUT/source-before.json"
"$REPO/vllm/flashnext/harness/x5_serve_proxy.sh" > "$OUT/serve.log" 2>&1 &
SPID=$!
( while kill -0 "$SPID" 2>/dev/null; do
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if ((available < 8388608)); then
      if [[ "$(docker inspect --format '{{.State.Running}}' "$NAME" 2>/dev/null || true)" == true ]]; then
        echo "ABORT_LOW_RAM $available"
        docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
        break
      fi
    fi
    sleep 2
  done ) &
WPID=$!
cleanup() {
  kill "$WPID" 2>/dev/null || true
  wait "$WPID" 2>/dev/null || true
  docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for ((i=0;i<7200;i++)); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"; then ready=1; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 3
done
ALIAS=qwen3.6-35b-a3b-${QUANT,,}-x5-mtp${MTP_TOKENS}
if ((ready)); then
  if ! python3 - "$OUT/models.json" "$ALIAS" <<'PY'
import json, sys
ids = [row['id'] for row in json.load(open(sys.argv[1]))['data']]
assert ids[0] == 'hotschmoe-dd' and sys.argv[2] in ids, ids
print('IDENTITY_EXACT_ALIAS_OK', sys.argv[2])
PY
  then ready=0; echo IDENTITY_FAILED; fi
fi
if ((ready)); then
  if [[ "${B70_X5_GDN_BF16:-0}" == 1 ]]; then
    if grep -q 'b70_x5_d3 gdn fused call' "$OUT/serve.log"; then
      echo FUSED_GDN_ACTIVE=1
    else
      echo FUSED_GDN_ACTIVE=0
    fi
  fi
  PY=python3
  # Only the requests hold CPU quiet; never hold it while queued for a GPU.
  set +e
  "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
    --url "http://127.0.0.1:$PORT" --out "$OUT/greedy.json" --respect-eos
  echo GREEDY_RC=$?
  "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
    --url "http://127.0.0.1:$PORT" --out "$OUT/repeat.json" --respect-eos --baseline "$OUT/greedy.json"
  echo REPEAT_RC=$?
  "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/bench_latency.py" \
    --base-url "http://127.0.0.1:$PORT" --model hotschmoe-dd --alias "$ALIAS" \
    --out "$OUT/bench.json" --decode-tokens 128 --skip-prefill --tag "X5-$TAG"
  echo BENCH_RC=$?
  if [[ "$MTP_TOKENS" == 0 ]]; then
    "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x5_quality.py" \
      --url "http://127.0.0.1:$PORT" --out "$OUT/quality.json"
    echo QUALITY_RC=$?
    STOCK=${X5_COMPARE_BASELINE:-$B/x5/${QUANT,,}-mtp0-gdn0-ev${B70_X5_EVENTLESS:-0}/greedy.json}
    if [[ "${B70_X5_GDN_BF16:-0}" == 1 && -f "$STOCK" ]]; then
      "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
        --url "http://127.0.0.1:$PORT" --out "$OUT/stock-parity.json" --respect-eos --baseline "$STOCK"
      echo STOCK_PARITY_RC=$?
    fi
  else
    BASE=${X5_BASELINE:-$B/x5/${QUANT,,}-mtp0-gdn${B70_X5_GDN_BF16:-0}-ev${B70_X5_EVENTLESS:-0}/greedy.json}
    if [[ -f "$BASE" ]]; then
      "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
        --url "http://127.0.0.1:$PORT" --out "$OUT/parity.json" --respect-eos --baseline "$BASE"
      echo PARITY_RC=$?
    fi
  fi
  curl -sf "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics.txt"
  python3 "$REPO/vllm/flashnext/harness/x5_manifest.py" "$OUT/source-after.json"
  set -e
fi
cleanup
trap - EXIT
"$REPO/bin/gpu-run" --card "$CARD" "$REPO/bin/xpu-health" --card "$CARD" > "$OUT/health-after.txt" 2>&1
echo "X5_WINDOW_DONE ready=$ready"
grep -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
test "$ready" = 1
