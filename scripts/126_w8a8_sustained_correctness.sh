#!/usr/bin/env bash
# 126_w8a8_sustained_correctness.sh -- prove W8A8 TP=2 (with --skip-server-warmup, the coherence fix)
# is CORRECT under SUSTAINED MIXED prefill+decode load -- the exact agentic pattern that makes vLLM emit
# "!!!!". Single-request coherence does NOT prove the GDN state won't progressively poison; this does.
# Runs the validated gdn_nan_repro/dd_mixload, then a POST-load coherence probe (catches global poisoning).
# Holds both cards -> run under: ./bin/gpu-run bash scripts/126_w8a8_sustained_correctness.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:woq; NAME=sglang_w8a8; PORT=30000; SERVED=qwen36-27b-w8a8-sqgptq
CKPT=/models/Qwen3.6-27B-W8A8-sqgptq
SCRATCH="${SCRATCH:-/tmp/claude-1000/-mnt-vm-8tb-github-b70-ai-things/dac41d4b-4204-4d31-bd5d-f4f587c287f6/scratchpad}/gdn_w8a8"
LOG=$REPO/sglang/w8a8_sustained.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# patched probe dir: backhoe model -> our W8A8 served id (Model Identity rule keeps the serve id correct)
mkdir -p "$SCRATCH"
cp "$REPO"/contrib/gdn_nan_repro/dd_mixload.py "$SCRATCH"/
python3 -c "import json; d=json.load(open('$REPO/contrib/gdn_nan_repro/backhoe_req.json')); d['model']='$SERVED'; json.dump(d, open('$SCRATCH/backhoe_req.json','w'))"

say "=== serve W8A8 TP=2 (skip-warmup) for sustained-load correctness ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_W8A8=1 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --tp 2 --context-length 8192 --mem-fraction-static 0.90 --skip-server-warmup \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health..."
ok=0
for i in $(seq 1 120); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$LOG"; ok=2; break; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "serve not healthy; abort"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }

coh(){ # tag
  local g; g=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France in one word.\"}],\"max_tokens\":24,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json;d=json.load(sys.stdin);print('COHERENCE[$1]:',repr(d['choices'][0]['message']['content'][:120]))" | tee -a "$LOG" || say "coh[$1] parse fail: $g"
}
say "=== PRE-load coherence ==="; coh pre

say "=== SUSTAINED MIXED LOAD (dd_mixload: 6 anchors + 6x12 bursts every 2s) ==="
python3 "$SCRATCH/dd_mixload.py" "$PORT" 6 6 12 2.0 400 1200 2>&1 | tee -a "$LOG"

say "=== POST-load coherence (detects progressive/global GDN poisoning) ==="; coh post

say "=== DONE. stopping. ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped $NAME"
