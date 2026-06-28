#!/usr/bin/env bash
# 139_xpu_cudagraph.sh -- FRONTIER: make sglang CAPTURE a decode XPUGraph on B70 (break the ~9.4 eager
# ceiling). int4 NO-MTP, single bs=1 bucket. Mounts xpu_cudagraph.py (device-gate + XPUAttentionBackend
# decode graph hooks) + the updated woq_shim.py (which installs it under B70_XPU_CUDAGRAPH=1).
# Gates: (1) CAPTURE "cuda graph: True"; (2) COHERENCE (not "!!!!"); (3) SPEEDUP vs 9.4; (4) SOAK stability.
#   ./bin/gpu-run --card 0 bash scripts/139_xpu_cudagraph.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_xpucg; PORT=30000; SERVED=qwen36-27b-int4-xpucg
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
# ATTN=intel_xpu (XPU FlashAttn) hits the SYCL-Graph work_group_scratch_memory wall at capture; ATTN=triton
# (pure-triton, no SYCL scratch feature + its own graph hooks) is the documented workaround (== vLLM TRITON_ATTN).
ATTN="${ATTN:-triton}"
SP=/opt/venv/lib/python3.12/site-packages
LOG="$REPO/sglang/xpu_cudagraph.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== int4 NO-MTP + XPUGraph decode capture (bs=1 bucket) ==="
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
    --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 \
    --max-running-requests 1 \
    --tp 1 --context-length 4096 --mem-fraction-static 0.90 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (graph capture happens at startup -> may be slow or CRASH)..."
ok=0
for i in $(seq 1 120); do
  docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "*** CONTAINER EXITED (capture crash?) ***"; docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|error|assert|graph|capture|xpu|sycl|level.zero|runtime|exception|Traceback|File \"" | tail -30 | tee -a "$LOG"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; log tail:"; docker logs "$NAME" 2>&1|tail -30|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }

say "=== install messages ==="; docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|cudagraph ENABLED|init_cuda_graphs gate" | tee -a "$LOG"

say "=== GATE 2: coherence (also JITs/triggers capture on first decode) ==="
g=$(curl -s --max-time 240 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
echo "$g" | python3 -c "import sys,json
from collections import Counter
try: t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('GATE2 PARSE_FAIL'); raise SystemExit
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print('GATE2 coherence:', (v if t else 'EMPTY'), '::', repr(t[:90]))" | tee -a "$LOG"

say "=== GATE 1: did it CAPTURE? (look for 'cuda graph: True' in decode batches) ==="
docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"

say "=== GATE 3+4: warm bench + soak (only meaningful if captured + coherent) ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-xpucg" 2>&1 | tee -a "$LOG"
say "=== capture status after bench ==="; docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"
say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped"
