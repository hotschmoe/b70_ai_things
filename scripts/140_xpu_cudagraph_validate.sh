#!/usr/bin/env bash
# 140_xpu_cudagraph_validate.sh -- gate the XPUGraph decode-capture WIN (scripts/139: int4 no-MTP 23.5 t/s,
# 2.5x eager) from a single-stream result into a DRIVER. Captures MULTIPLE bs buckets (1,2,4 -> concurrency),
# reproduces c1, benches c4 aggregate, soak-stability, and -- the real risk -- SUSTAINED MIXED LOAD (does the
# GDN/mamba state stay correct under concurrent prefill+decode WITH graph replay? the agentic pattern that
# breaks vLLM). Sampling-capable (non-MTP), so the mixload uses the model's default sampling.
#   ./bin/gpu-run --card 0 bash scripts/140_xpu_cudagraph_validate.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_xpucg; PORT=30000; SERVED=qwen36-27b-int4-xpucg
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
ATTN="${ATTN:-triton}"; MAXREQ="${MAXREQ:-4}"; BSLIST="${BSLIST:-1 2 4}"; BSMAX="${BSMAX:-4}"
SP=/opt/venv/lib/python3.12/site-packages
LOG="$REPO/sglang/xpu_cudagraph_validate.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== int4 NO-MTP XPUGraph DRIVER validation: bs buckets [$BSLIST] maxreq=$MAXREQ attn=$ATTN ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$REPO/sglang/patches/woq_shim.py:$SP/woq_shim.py:ro" \
  -v "$REPO/sglang/patches/xpu_cudagraph.py:$SP/xpu_cudagraph.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_CUDAGRAPH=1 -e B70_XPU_CUDAGRAPH_DEBUG="${DBG:-1}" \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend $ATTN --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --cuda-graph-bs-decode $BSLIST --cuda-graph-max-bs-decode $BSMAX \
    --max-running-requests $MAXREQ \
    --tp 1 --context-length 4096 --mem-fraction-static 0.90 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (multi-bucket capture at startup -> slower/may CRASH)..."
ok=0
for i in $(seq 1 150); do
  docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "*** CONTAINER EXITED (multi-bs capture crash?) ***"; docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|error|assert|graph|capture|sycl|scratch|runtime|exception|Traceback|File \"|line [0-9]" | tail -30 | tee -a "$LOG"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; log tail:"; docker logs "$NAME" 2>&1|tail -30|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }
docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|hooks installed|init_cuda_graphs gate" | sort -u | tee -a "$LOG"

say "=== coherence (greedy) ==="
g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
echo "$g"|python3 -c "import sys,json
from collections import Counter
try: t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('PARSE_FAIL'); raise SystemExit
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print('coherence:', (v if t else 'EMPTY'), '::', repr(t[:90]))" | tee -a "$LOG"

say "=== SAMPLING check (this driver supports it, unlike MTP): temperature=0.8 should vary ==="
for k in 1 2; do
  s=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 6-word story about the sea.\"}],\"max_tokens\":40,\"temperature\":0.8,\"seed\":$k}")
  echo "$s"|python3 -c "import sys,json;print('  sample$k:',repr((json.load(sys.stdin)['choices'][0]['message']['content'] or '')[:80]))" 2>/dev/null | tee -a "$LOG" || say "  sample$k parse-fail"
done

say "=== GATE capture (cuda graph True/False counts) ==="
docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"

say "=== pp/ttft/tg regime (warm c1/c4 + soak) ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-xpucg-drv" 2>&1 | tee -a "$LOG"

say "=== SUSTAINED MIXED LOAD (GDN-under-graph-replay correctness -- the agentic pattern that breaks vLLM) ==="
PD="${SCRATCH:-/tmp/claude-1000/-mnt-vm-8tb-github-b70-ai-things/3d35fc3f-649b-4a06-b50e-ec2ce0215970/scratchpad}/xpucg_mix"
mkdir -p "$PD"; cp "$REPO"/contrib/gdn_nan_repro/dd_mixload.py "$PD"/
python3 -c "import json;d=json.load(open('$REPO/contrib/gdn_nan_repro/backhoe_req.json'));d['model']='$SERVED';json.dump(d,open('$PD/backhoe_req.json','w'))"
python3 "$PD/dd_mixload.py" "$PORT" 3 3 6 2.0 300 600 2>&1 | tee -a "$LOG"
upnow=$(docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -c "$NAME" || true)
say "post-mixload: container running=$upnow (0 = CRASHED under load)"
pg=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France in one word.\"}],\"max_tokens\":24,\"temperature\":0}")
echo "$pg"|python3 -c "import sys,json;print('POST-LOAD COHERENCE:',repr(json.load(sys.stdin)['choices'][0]['message']['content'][:80]))" | tee -a "$LOG" || say "post-load parse fail"
say "=== final capture counts ==="; docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"
say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped -> $LOG"
