#!/usr/bin/env bash
# code_ab_session.sh -- serve one TP=2 shelf entry at its best DD config and bench REAL-CODE decode.
# Runs the decode bench FIRST after load (fairest thermal state). Under the lease:
#   SHELF=<path> LABEL=<x> EXTRA="<env>" ./bin/gpu-run bash vllm/nvfp4/code_ab_session.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SHELF="${SHELF:?set SHELF=<rdy_to_serve/.../serve.sh>}"
LABEL="${LABEL:-ab}"; PORT="${PORT:-8079}"; NAME="${NAME:-ab_bench}"; KEY="${API_KEY:-abtestkey}"
MAXLEN="${MAXLEN:-32768}"; EXTRA="${EXTRA:-}"

echo "== [$LABEL] serve TP=2 best config: $SHELF (MAXLEN=$MAXLEN $EXTRA) =="
env $EXTRA TP=2 PORT="$PORT" NAME="$NAME" MAXLEN="$MAXLEN" API_KEY="$KEY" \
  bash "$SHELF" start >/tmp/claude-1000/serve_ab_$LABEL.log 2>&1 || { echo launch-failed; exit 1; }
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || { echo "died:"; docker logs --tail 40 "$NAME" 2>&1; exit 2; }
  sleep 5
done
[ "${ok:-0}" = 1 ] || { echo "health timeout"; docker logs --tail 60 "$NAME" 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; exit 3; }
AUTH=(-H "Authorization: Bearer $KEY")
SERVED=$(curl -s "${AUTH[@]}" "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "[$LABEL] served: $SERVED  (healthy ~$((i*5))s)"

echo "----- [$LABEL] REAL-CODE DECODE (c1, first after load) -----"
API_KEY="$KEY" python3 "$REPO/vllm/nvfp4/bench_code.py" "http://localhost:$PORT/v1" "$SERVED" 1 384 3
echo "----- [$LABEL] REAL-CODE DECODE (c2) -----"
API_KEY="$KEY" python3 "$REPO/vllm/nvfp4/bench_code.py" "http://localhost:$PORT/v1" "$SERVED" 2 384 2

docker rm -f "$NAME" >/dev/null 2>&1; echo "[$LABEL] torn down"
