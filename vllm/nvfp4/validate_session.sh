#!/usr/bin/env bash
# validate_session.sh -- coherence gate (18/18) + decode-parity check for the push-AR prefill overlay.
# Run under the lease:  PUSH_AR=1 ./bin/gpu-run bash vllm/nvfp4/validate_session.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
DIR="$REPO/vllm/nvfp4"
LABEL="${LABEL:-validate}"; PORT="${PORT:-8079}"; NAME="${NAME:-nvfp4_tp2_prefill}"
export TP=2 MODE=fused GRAPH=1 MTPTOK=5 CAPSIZES=1,2,4,8 MAXLEN="${MAXLEN:-16384}" UTIL=0.85 MAXSEQS=8
export PREFIXCACHE=1 PUSH_AR="${PUSH_AR:-0}" PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-65536}" NAME PORT

echo "== [$LABEL] serve (PUSH_AR=$PUSH_AR) =="
bash "$DIR/serve_nvfp4_27b.sh" >/tmp/claude-1000/serve_$LABEL.log 2>&1 || { echo "launch failed"; exit 1; }
for i in $(seq 1 144); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || { echo "container died:"; docker logs --tail 40 "$NAME"; exit 2; }
  sleep 5
done
[ "${ok:-0}" = 1 ] || { echo "health timeout"; docker logs --tail 60 "$NAME"; docker rm -f "$NAME" >/dev/null 2>&1; exit 3; }
SERVED=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "[$LABEL] served: $SERVED  (healthy ~$((i*5))s)"
[ "$PUSH_AR" = 1 ] && docker logs "$NAME" 2>&1 | grep -i 'ENGAGED' | tail -1

echo "== [$LABEL] CONCURRENT COHERENCE GATE (3 waves x 6 = 18 mixed prefill+decode) =="
python3 "$DIR/../gate_concurrent_coherence.py" "http://localhost:$PORT/v1" "$SERVED" 3 6 200
GATE=$?
echo "== [$LABEL] DECODE-PARITY BENCH (IN=2048 OUT=128, warm; TG/stream must match baseline) =="
python3 "$DIR/bench_2048.py" "http://localhost:$PORT/v1" "$SERVED" 4 128

docker rm -f "$NAME" >/dev/null 2>&1
echo "[$LABEL] torn down; GATE exit=$GATE (0=PASS)"
exit $GATE
