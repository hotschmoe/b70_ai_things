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
( owned_start=0
  while kill -0 "$SPID" 2>/dev/null; do
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ "$(docker inspect --format '{{.State.Running}}' "$NAME" 2>/dev/null || true)" == true ]]; then
      if ((owned_start == 0)); then owned_start=$SECONDS; fi
      if ((available < 8388608)); then
        echo "ABORT_LOW_RAM $available"
        docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
        break
      fi
      if ((SECONDS - owned_start > ${X5_MAX_WINDOW_SECONDS:-600})); then
        echo ABORT_WINDOW_DEADLINE
        docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
        break
      fi
    fi
    sleep 2
  done ) &
WPID=$!
cleanup() {
  if [[ "${cleaned:-0}" == 1 ]]; then return 0; fi
  cleaned=1
  kill "$WPID" 2>/dev/null || true
  wait "$WPID" 2>/dev/null || true
  docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
  if wait "$SPID"; then serve_rc=0; else serve_rc=$?; fi
  echo "$serve_rc" > "$OUT/serve-exit-code.txt"
  echo "SERVE_EXIT_CODE=$serve_rc"
}
trap cleanup EXIT
gate_rc=0
record_rc() {
  echo "${1}_RC=$2"
  if [[ "$2" != 0 ]]; then gate_rc=$2; fi
  return 0
}
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
  record_rc GREEDY "$?"
  "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
    --url "http://127.0.0.1:$PORT" --out "$OUT/repeat.json" --respect-eos --baseline "$OUT/greedy.json"
  record_rc REPEAT "$?"
  "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/bench_latency.py" \
    --base-url "http://127.0.0.1:$PORT" --model hotschmoe-dd --alias "$ALIAS" \
    --out "$OUT/bench.json" --decode-tokens 128 --skip-prefill --tag "X5-$TAG"
  record_rc BENCH "$?"
  if [[ "$MTP_TOKENS" == 0 ]]; then
    "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x5_quality.py" \
      --url "http://127.0.0.1:$PORT" --out "$OUT/quality.json"
    record_rc QUALITY "$?"
    STOCK=${X5_COMPARE_BASELINE:-$B/x5/${QUANT,,}-mtp0-gdn0-ev${B70_X5_EVENTLESS:-0}/greedy.json}
    if [[ "${B70_X5_GDN_BF16:-0}" == 1 && -f "$STOCK" ]]; then
      "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
        --url "http://127.0.0.1:$PORT" --out "$OUT/stock-parity.json" --respect-eos --baseline "$STOCK"
      record_rc STOCK_PARITY "$?"
    fi
  else
    BASE=${X5_BASELINE:-$B/x5/${QUANT,,}-mtp0-gdn${B70_X5_GDN_BF16:-0}-ev${B70_X5_EVENTLESS:-0}/greedy.json}
    if [[ -f "$BASE" ]]; then
      "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x1_compare_greedy.py" \
        --url "http://127.0.0.1:$PORT" --out "$OUT/parity.json" --respect-eos --baseline "$BASE"
      record_rc PARITY "$?"
    else
      echo PARITY_MISSING_BASELINE
      gate_rc=3
    fi
  fi
  if [[ "${X5_LMX:-0}" == 1 ]]; then
    "$REPO/bin/cpu-run" --quiet "$PY" "$REPO/vllm/flashnext/harness/x5_lmx_local.py" "$OUT"
    record_rc LOCALMAXXING "$?"
  fi
  curl -sf "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics.txt"
  python3 "$REPO/vllm/flashnext/harness/x5_manifest.py" "$OUT/source-after.json"
  python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
a = json.loads((p / 'source-before.json').read_text())['sha256']
b = json.loads((p / 'source-after.json').read_text())['sha256']
drift = [k for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]
print('SOURCE_DRIFT', json.dumps(drift, ensure_ascii=True))
raise SystemExit(3 if drift else 0)
PY
  record_rc SOURCE "$?"
  set -e
fi
cleanup
trap - EXIT
"$REPO/bin/gpu-run" --card "$CARD" "$REPO/bin/xpu-health" --card "$CARD" > "$OUT/health-after.txt" 2>&1
echo "X5_WINDOW_DONE ready=$ready serve_rc=$serve_rc gate_rc=$gate_rc"
grep -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
test "$ready" = 1
if [[ "$serve_rc" != 0 ]]; then exit "$serve_rc"; fi
exit "$gate_rc"
