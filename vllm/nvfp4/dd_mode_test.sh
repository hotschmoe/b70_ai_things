#!/usr/bin/env bash
# dd_mode_test.sh -- validate the NVFP4 TP=2 daily-driver mode END TO END via the shelf serve.sh:
# parsers (tool-call + reasoning), API-key enforcement, a fresh decode number, coherence.
#   API_KEY=testkey123 ./bin/gpu-run bash vllm/nvfp4/dd_mode_test.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh"
PORT="${PORT:-8079}"; NAME="${NAME:-nvfp4_27b}"; export NAME PORT
KEY="${API_KEY:-testkey123}"; export API_KEY="$KEY"
export TP=2 MAXLEN="${MAXLEN:-131072}"   # 128K min; keep modest for a fast test

echo "== serve TP=2 DD mode (push-AR+MTP+prefixcache+parsers, key enforced) =="
TP=2 API_KEY="$KEY" MAXLEN="$MAXLEN" PORT="$PORT" NAME="$NAME" bash "$SHELF" start >/tmp/claude-1000/serve_ddtest.log 2>&1 || { echo launch-failed; exit 1; }
for i in $(seq 1 144); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || { echo "died:"; docker logs --tail 40 "$NAME"; exit 2; }
  sleep 5
done
[ "${ok:-0}" = 1 ] || { echo "health timeout"; docker logs --tail 60 "$NAME"; docker rm -f "$NAME"; exit 3; }
AUTH=(-H "Authorization: Bearer $KEY")
SERVED=$(curl -s "${AUTH[@]}" "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "served: $SERVED"
docker logs "$NAME" 2>&1 | grep -iE 'ENGAGED|tool.call.parser|reasoning|api.key|GPU KV cache size' | tail -6

echo "== 1) API-key ENFORCED on /v1/* (no key -> 401) =="
code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/v1/models"); echo "  no-key /v1/models -> HTTP $code (expect 401)"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health"); echo "  no-key /health    -> HTTP $code (expect 200, open)"

echo "== 2) TOOL CALL (qwen3_coder parser -> structured tool_calls) =="
python3 - "$PORT" "$SERVED" "$KEY" <<'PY'
import sys,json,urllib.request
port,model,key=sys.argv[1],sys.argv[2],sys.argv[3]
body=json.dumps({"model":model,"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"auto","max_tokens":256,"temperature":0.0}).encode()
req=urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",data=body,
  headers={"content-type":"application/json","Authorization":f"Bearer {key}"})
r=json.load(urllib.request.urlopen(req,timeout=120)); m=r["choices"][0]["message"]
tc=m.get("tool_calls")
if tc: print("  TOOL_CALLS OK:",tc[0]["function"]["name"],tc[0]["function"]["arguments"])
else:  print("  NO tool_calls; content=",repr((m.get('content') or '')[:120]))
print("  reasoning_content present:", bool(m.get("reasoning_content")))
PY

echo "== 3) FRESH decode bench (IN=2048 OUT=128, first thing after load = clean clocks) =="
API_KEY="$KEY" python3 "$REPO/vllm/nvfp4/bench_2048.py" "http://localhost:$PORT/v1" "$SERVED" 1 128

docker rm -f "$NAME" >/dev/null 2>&1; echo "torn down"
