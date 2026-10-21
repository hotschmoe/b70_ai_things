#!/bin/bash
# F1 copy of O4 serve window: O4 decode recipe + F1 prefill delta, F1 probe.
# health -> vllm serve (recipe flags from o4_recipe.env) -> /v1/models
# identity -> greedy 17*19 / 23*29 -> bench_latency (cpu-run --quiet)
# -> stop -> health. Copy this script before queueing it.
#
#   TAG=s1 bash vllm/flashnext/harness/o4_serve.sh
#
# Env: TAG, PORT (18147), MAX_LEN (2048), DECODE (64), PREFILL (1024),
# O4_RECIPE (env file, default harness/o4_recipe.env), O4_SERVE_EXTRA
# ("K=V ..." appended), SKIP_HEALTH_BEFORE=1, SERVE_ARGS_EXTRA.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
H=$REPO/vllm/flashnext/harness
B=/mnt/vm_8tb/b70/build/vllm-flashnext
TAG="${TAG:?TAG}"
PORT="${PORT:-18147}"
OUT="$B/f1/merged-$TAG"
NAME="vllm-fn-f1-merged-$TAG"
HARD_KB=8388608
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window.log") 2>&1
echo "=== O4 serve $TAG $(date -u +%H:%M:%S) ==="
RECIPE="${O4_RECIPE:-$H/o4_recipe.env}"
ENVS=()
DELTA="${F1_DELTA:-$H/f1_prefill_delta.env}"
for kv in $(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | xargs) $(grep -v '^#' "$DELTA" | sed 's/ *#.*//' | xargs) ${O4_SERVE_EXTRA:-}; do ENVS+=(--env "$kv"); done
GGUF_SO=$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep '^B70_GGUF_SO=' | cut -d= -f2)
sha256sum "$GGUF_SO" "$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep '^B70_EP_SO=' | cut -d= -f2)" \
  "$REPO/vllm/flashnext/plugins/b70_o4_xpu/b70_o4_xpu/_C.so" | cut -c1-16
AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "MEM_AVAILABLE_KB:$AVAIL_KB"
# Another agent's load can hold the RAM; wait up to 2 h for 60 GB.
for _w in $(seq 1 720); do
  AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  [ "$AVAIL_KB" -ge 60000000 ] && break
  [ $((_w % 30)) = 1 ] && echo "waiting for RAM: MemAvailable_kB=$AVAIL_KB $(date -u +%H:%M:%S)"
  sleep 10
done
if [ "$AVAIL_KB" -lt 60000000 ]; then
  echo "ABORT MemAvailable under 60 GB before a Flash-Next load"; echo WINDOW_DONE; exit 4
fi
if [ "${SKIP_HEALTH_BEFORE:-0}" != 1 ]; then
  B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=14400 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-before.txt" 2>&1
  echo "HEALTH_BEFORE_RC:$?"
  grep -q "xpu-health: HEALTHY" "$OUT/health-before.txt" || { echo "health before failed"; echo WINDOW_DONE; exit 3; }
fi
PYPATH="$B/c7b-fa-overlay:$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_ep_xpu:$REPO/vllm/flashnext/plugins/b70_qsa_xpu:$REPO/vllm/flashnext/plugins/b70_o4_xpu"
CFG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,8]}'
( while true; do a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$a" -lt "$HARD_KB" ]; then echo "ABORT_LOW_RAM $a"; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 0; fi
    sleep 2; done ) &
WD=$!
B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=14400 timeout --signal=TERM --kill-after=30 40m \
"$H/xpu-run.sh" --card both --memory 100g --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 --env B70_EP=1 --env B70_QSA=0 \
  --env "B70_GGUF_PATH=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
  --env "B70_EP_PROFILE=/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin" \
  --env B70_EP_Q0=138 --env B70_EP_SPLIT=profile \
  --env B70_PLE_NBYTES=28800138240 --env B70_EP_MEM_MIN_KB=$HARD_KB --env B70_EP_MAX_T=8192 --env "B70_EP_PREFILL_CTL=$OUT/ctl.txt" \
  --env B70_XPU_CG_RECLAIM=500 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  --env "PYTHONPATH=$PYPATH" --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 --env VLLM_ENGINE_READY_TIMEOUT_S=1200 \
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-o4" \
  "${ENVS[@]}" \
  -- vllm serve /mnt/vm_8tb/b70/build/backend-decision/hf-flashnext \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name hotschmoe-dev qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4 \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len "${MAX_LEN:-2048}" \
  --max-num-batched-tokens "${MAX_BATCHED:-12288}" --max-num-seqs 4 --gpu-memory-utilization 0.98 \
  --language-model-only --load-format b70_gguf --quantization b70_gguf --trust-remote-code \
  --attention-backend FLASH_ATTN --no-enable-prefix-caching --mamba-cache-mode align \
  --compilation-config "$CFG" ${SERVE_ARGS_EXTRA:-} > "$OUT/serve.log" 2>&1 &
SP=$!
ready=0
for i in $(seq 1 360); do
  kill -0 $SP 2>/dev/null || { echo "serve exited before ready"; break; }
  # Ready only when THIS server answers (alias check; another agent may share the port).
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" \
     && grep -q "qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4" "$OUT/models.json"; then ready=1; echo "READY $(date -u +%H:%M:%S)"; break; fi
  sleep 5
done
GREEDY_RC=2; BENCH_RC=2
if [ $ready = 1 ]; then
  awk '/MemAvailable:|AnonPages:/{print}' /proc/meminfo > "$OUT/rss-ready.txt"
  PORT=$PORT OUT=$OUT python3 - <<'PY'
import json, os, urllib.request
port = os.environ["PORT"]; out = os.environ["OUT"]
ids = [m["id"] for m in json.load(open(f"{out}/models.json"))["data"]]
rows = [{"ids": ids, "first_ok": ids[:2] == ["hotschmoe-dev", "qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4"]}]
for content, ref in (("Return only the integer: 17 * 19.", "323"), ("Return only the integer: 23 * 29.", "667")):
    body = json.dumps({"model": "hotschmoe-dev", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    rows.append({"ref": ref, "text": text, "match": text.strip() == ref})
json.dump(rows, open(f"{out}/greedy.json", "w"), indent=2)
print("GREEDY", json.dumps(rows))
raise SystemExit(0 if rows[0]["first_ok"] and all(r["match"] for r in rows[1:]) else 1)
PY
  GREEDY_RC=$?
  echo "GREEDY_RC:$GREEDY_RC"
  if [ $GREEDY_RC = 0 ]; then
    B70_AGENT=F1 B70_CPU_LOCK_TIMEOUT=900 "$REPO/bin/cpu-run" --quiet python3 "$H/bench_latency.py" \
      --base-url "http://127.0.0.1:$PORT" --model hotschmoe-dev --alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4 \
      --decode-tokens "${DECODE:-64}" --prefill "${PREFILL:-1024}" --tag "f1-merged-$TAG" --timeout 600 --out "$OUT/bench.json"
    BENCH_RC=$?
    F1_BASE="http://127.0.0.1:$PORT" F1_OUT="$OUT/probe.json" F1_ALIAS=qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4 \
      F1_LENGTHS="${F1_LENGTHS:-1024,8192,12288,32768}" F1_REPEAT="${F1_REPEAT:-1}" F1_DECODE="${F1_DECODE:-1}" \
      F1_COHERENCE_LEN="${F1_COHERENCE_LEN:-32768}" "$REPO/bin/cpu-run" --quiet /usr/bin/python3 "$H/f1_probe.py" > "$OUT/probe.log" 2>&1
    echo "PROBE_RC:$? $(date -u +%H:%M:%S)"; grep -E "^F1 " "$OUT/probe.log" | sed 's/"usage.*//' | cut -c1-170
  fi
fi
echo "BENCH_RC:$BENCH_RC"
docker stop -t 20 "$NAME" >/dev/null 2>&1 || true
wait $SP; echo "SERVE_RC:$?"
kill $WD 2>/dev/null
grep -E "PLE device|mlock|fused into|Traceback|Error" "$OUT/serve.log" | tail -n 8
B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=1800 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 2 "$OUT/health-after.txt"
echo WINDOW_DONE
