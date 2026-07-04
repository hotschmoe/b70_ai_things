#!/usr/bin/env bash
# prefill_session.sh -- one gpu-run-wrapped session for the NVFP4 TP=2 prefill campaign (Track 11g).
# Serves the NVFP4 27B TP=2 config (env passthrough to serve_nvfp4_27b.sh), waits healthy, runs the
# cold-prefill bench + a coherence probe, then tears down. Hold the GPU lease for the WHOLE run:
#     ./bin/gpu-run bash vllm/nvfp4/prefill_session.sh
# Config knobs (env): LABEL, PORT, and any serve_nvfp4_27b.sh env (PUSH_AR_* to enable the overlay).
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
DIR="$REPO/vllm/nvfp4"

LABEL="${LABEL:-baseline}"
PORT="${PORT:-8079}"
NAME="${NAME:-nvfp4_tp2_prefill}"
LENS="${LENS:-512,2048,8192}"
KEEP="${KEEP:-0}"   # 1 = leave container up after bench (for manual poking)

# Campaign default config: TP=2 fused GRAPH=1 MTP5, matches the DD except MAXLEN (smaller = faster boot,
# still fits IN=8192 tests) and PREFIXCACHE (unique prompts make cache always miss = cold either way).
export TP="${TP:-2}" MODE="${MODE:-fused}" GRAPH="${GRAPH:-1}" MTPTOK="${MTPTOK:-5}"
export CAPSIZES="${CAPSIZES:-1,2,4,8}" MAXLEN="${MAXLEN:-16384}" UTIL="${UTIL:-0.85}"
export MAXSEQS="${MAXSEQS:-8}" PREFIXCACHE="${PREFIXCACHE:-1}"
export PUSH_AR="${PUSH_AR:-0}" PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-65536}"
export NAME PORT

echo "=================================================================="
echo "[$LABEL] serving NVFP4 TP=$TP MODE=$MODE GRAPH=$GRAPH MTP=$MTPTOK MAXLEN=$MAXLEN"
echo "        PUSH_AR_SO=${PUSH_AR_SO:-<unset>} PUSH_AR_MIN_NUMEL=${PUSH_AR_MIN_NUMEL:-<unset>}"
echo "=================================================================="

bash "$DIR/serve_nvfp4_27b.sh" >/tmp/claude-1000/serve_$LABEL.log 2>&1 || { echo "serve.sh launch failed"; exit 1; }

# wait healthy (up to 12 min: model load + capture)
echo "[$LABEL] waiting for /health ..."
ok=0
for i in $(seq 1 144); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  # bail early if the container died
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME\$"; then
    echo "[$LABEL] container $NAME exited during startup:"; docker logs --tail 40 "$NAME" 2>&1; exit 2
  fi
  sleep 5
done
[ "$ok" = 1 ] || { echo "[$LABEL] health TIMEOUT"; docker logs --tail 60 "$NAME" 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; exit 3; }
echo "[$LABEL] healthy after ~$((i*5))s"

SERVED=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[$LABEL] served model id: $SERVED"

# confirm push-AR engaged (if configured)
if [ "${PUSH_AR:-0}" = 1 ]; then
  echo "[$LABEL] push_ar log lines:"; docker logs "$NAME" 2>&1 | grep -i 'push.ar\|push all' | tail -8
fi

echo "----- [$LABEL] COLD PREFILL BENCH (c1) -----"
python3 "$DIR/bench_prefill.py" "http://localhost:$PORT/v1" "$SERVED" 1 8 "$LENS" 3
echo "----- [$LABEL] COLD PREFILL BENCH (c4) -----"
python3 "$DIR/bench_prefill.py" "http://localhost:$PORT/v1" "$SERVED" 4 8 "$LENS" 2

echo "----- [$LABEL] coherence probe -----"
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, urllib.request
port, model = sys.argv[1], sys.argv[2]
for q in ["What is 17+26? Reply with just the number.",
          "Name the color of a clear daytime sky in one word."]:
    body=json.dumps({"model":model,"messages":[{"role":"user","content":q}],
                     "max_tokens":512,"temperature":0.0}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,
                               headers={"content-type":"application/json"})
    try:
        r=json.load(urllib.request.urlopen(req,timeout=120))
        print("  Q:",q,"-> A:",repr(r["choices"][0]["message"]["content"].strip()[:60]))
    except Exception as e:
        print("  Q:",q,"-> ERROR",e)
PY

if [ "$KEEP" = 1 ]; then
  echo "[$LABEL] KEEP=1, leaving $NAME up on port $PORT"
else
  docker rm -f "$NAME" >/dev/null 2>&1
  echo "[$LABEL] torn down"
fi
