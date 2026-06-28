#!/usr/bin/env bash
# 143_graph_mtp_stack.sh -- FRONTIER++: stack XPUGraph decode capture + NEXTN MTP. If MTP's token amortization
# (accept_len~4.4) rides on the graph's launch collapse, single-stream could beat 23.5 (graph no-MTP) and 15.3
# (eager MTP). Opens the EAGLE draft cuda-graph device gate to xpu (xpu_cudagraph.py section 3) + uses ATTN=triton
# (triton has spec cuda-graph hooks: target_verify + draft_extend). Single card 0 (the fast compute card).
#   ./bin/gpu-run --card 0 bash scripts/143_graph_mtp_stack.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_gmtp; PORT=30000; SERVED=qwen36-27b-int4-gmtp
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
SP=/opt/venv/lib/python3.12/site-packages
MEMFRAC="${MEMFRAC:-0.88}"; SPEC_STEPS="${SPEC_STEPS:-7}"; SPEC_DRAFT="${SPEC_DRAFT:-8}"
LOG="$REPO/sglang/graph_mtp_stack.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== graph+MTP STACK: int4-mtp + B70_XPU_MTP=1 + B70_XPU_CUDAGRAPH=1 + ATTN=triton, NEXTN steps=$SPEC_STEPS ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$REPO/sglang/patches/woq_shim.py:$SP/woq_shim.py:ro" \
  -v "$REPO/sglang/patches/xpu_cudagraph.py:$SP/xpu_cudagraph.py:ro" \
  -v "$REPO/sglang/patches/mtp_tree_xpu.py:$SP/mtp_tree_xpu.py:ro" \
  -v "$REPO/sglang/patches/memory_pool.py:$SP/sglang/srt/mem_cache/memory_pool.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_CUDAGRAPH=1 -e B70_XPU_CUDAGRAPH_DEBUG=1 -e B70_MTP_DEBUG="${MDBG:-0}" \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend triton --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton \
    --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --max-running-requests 1 --skip-server-warmup \
    --tp 1 --context-length 4096 --mem-fraction-static $MEMFRAC --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (draft+verify+decode capture at startup -> slow or CRASH)..."
ok=0
for i in $(seq 1 180); do
  docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "*** CONTAINER EXITED (capture crash) ***"; docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|mtp|EAGLE|error|assert|graph|capture|sycl|scratch|KeyError|Traceback|File \"|line [0-9]|NotImplemented|spec" | tail -40 | tee -a "$LOG"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; tail:"; docker logs "$NAME" 2>&1|tail -40|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }
say "=== install messages ==="; docker logs "$NAME" 2>&1 | grep -iE "xpu-cudagraph|mtp-tree-xpu|EAGLE draft|gate:" | sort -u | tee -a "$LOG"

say "=== GATE 2: coherence (greedy; JITs spec+capture) ==="
g=$(curl -s --max-time 240 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
running=$(docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -c "$NAME" || true)
[ "$running" = 0 ] && { say "*** CRASHED during first gen ***"; docker logs "$NAME" 2>&1|grep -iE "error|traceback|assert|graph|spec|capture|File \"|line [0-9]"|tail -30|tee -a "$LOG"; exit 3; }
echo "$g"|python3 -c "import sys,json
from collections import Counter
try: t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('GATE2 PARSE_FAIL'); raise SystemExit
v='OK'
if len(t)>=16:
 c,n=Counter(t).most_common(1)[0]
 if n/len(t)>=0.6: v='GARBAGE'
print('GATE2 coherence:', (v if t else 'EMPTY'), '::', repr(t[:90]))" | tee -a "$LOG"

say "=== GATE 1: capture? (cuda graph True on verify/decode batches) ==="
docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"
say "=== GATE 3: accept length (spec math intact under capture?) ==="
docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | tail -8 | tee -a "$LOG"
docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | awk -F': ' '{s+=$2;n++} END{if(n)printf "[mean accept_len %d batches] %.2f\n",n,s/n}' | tee -a "$LOG"

say "=== GATE 4+5: warm c1 + soak (vs graph 23.5 / eager-MTP 15.3) ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-graph-mtp" 2>&1 | tee -a "$LOG"
say "=== final capture + accept ==="; docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c | tee -a "$LOG"
say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped -> $LOG"
