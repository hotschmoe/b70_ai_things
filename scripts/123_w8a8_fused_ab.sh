#!/usr/bin/env bash
# 123_w8a8_fused_ab.sh -- W8A8 TP=2 serve+bench with the FUSED int8 hybrid (vs legacy _int_mm).
# Adds the built int8 oneDNN ops (int8_gemm_w8a16 decode / int8_gemm_w8a8 prefill) to the W8A8 path.
# Defaults to the VISION ckpt (vision retained). Mirrors scripts/122 but with the fused .so + flags.
#
# Holds BOTH cards -> run under: ./bin/gpu-run bash scripts/123_w8a8_fused_ab.sh
# Knobs:
#   FUSED=1  (default)  B70_XPU_W8A8_FUSED=1 -> new hybrid ; FUSED=0 -> legacy torch._int_mm chain
#   GRAPH=1             B70_XPU_CUDAGRAPH=1 -> XPUGraph decode capture (HIGHER wedge risk; eager default)
#   CKPT, SERVED, CTX, MEMFRAC, COHERENCE_ONLY
set -uo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="${IMG:-sglang-xpu:woq}"
NAME="${NAME:-sglang_w8a8}"
CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-vision}"
SERVED="${SERVED:-qwen36-27b-w8a8-sqgptq-fused}"
FUSED="${FUSED:-1}"
GRAPH="${GRAPH:-0}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
PORT=30000
TP=2
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
COHERENCE_ONLY="${COHERENCE_ONLY:-0}"
EXTRA_FLAGS="--skip-server-warmup"   # REQUIRED for W8A8 GDN coherence (scripts/122 root cause)
[ "$GRAPH" = 1 ] && EXTRA_FLAGS="$EXTRA_FLAGS"   # graph flags handled via env (woq_shim)
TOK="/models/Qwen_Qwen3.6-27B"
LABEL="$([ "$FUSED" = 1 ] && echo fused || echo legacy)$([ "$GRAPH" = 1 ] && echo +graph)"
LOG="$REPO/w8a8/w8a8_fused_ab_${LABEL}.log"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"

# pre-flight health (TP=2 wedge discipline)
say "=== pre-flight xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG" || { say "UNHEALTHY pre-serve -- aborting"; exit 3; }

say "=== W8A8 TP=2 serve [$LABEL]: $SERVED  img=$IMG ctx=$CTX memfrac=$MEMFRAC ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_W8A8=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  $( [ "$FUSED" = 1 ] && echo "-e B70_XPU_W8A8_FUSED=1" ) \
  $( [ "$GRAPH" = 1 ] && echo "-e B70_XPU_CUDAGRAPH=1" ) \
  ${DENV:+$(for kv in $DENV; do echo -n "-e $kv "; done)} \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC \
    $EXTRA_FLAGS --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (model load ~3-5min)..."
ok=0
for i in $(seq 1 120); do
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    say "CONTAINER EXITED during load -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = 200 ]; then ok=1; say "/health 200 after ~$((i*5))s"; break; fi
  if docker logs "$NAME" 2>&1 | grep -qiE "coredumps before exiting|Scheduler hit an exception|Received sigquit|DEVICE_LOST"; then
    say "WORKER CRASHED -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break
  fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "SERVE NOT HEALTHY (ok=$ok). Tailing logs:"; docker logs "$NAME" 2>&1 | tail -60 | tee -a "$LOG"
  say "stopping container."; docker rm -f "$NAME" >/dev/null 2>&1
  say "=== post-teardown health check ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"
  exit 1
fi

say "=== shim wiring (count of fused layers / install line) ==="
docker logs "$NAME" 2>&1 | grep -E "w8a8-shim|w8a8-fused" | tail -5 | tee -a "$LOG"

say "=== coherence check (must NOT be '!!!!') ==="
gen=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Answer in two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
echo "$gen" | python3 -c "import sys,json; d=json.load(sys.stdin); print('COHERENCE:', repr(d['choices'][0]['message']['content'][:240]))" | tee -a "$LOG" || say "coherence parse failed: $gen"

if [ "$COHERENCE_ONLY" = 1 ]; then
  say "=== COHERENCE_ONLY: stopping. ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped"; exit 0
fi

bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
  --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
  --random-input-len ${3:-2048} --random-output-len ${4:-128} --num-prompts $2 --max-concurrency $1 2>&1"; }
report(){ local raw="$2" ttft tpot otps reqs dec pp
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  tpot=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  otps=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  reqs=$(echo "$raw"|grep -i 'Request throughput'|grep -oE '[0-9.]+'|head -1)
  dec=$(awk -v t="$tpot" 'BEGIN{if(t>0)printf"%.2f",1000.0/t;else print"NA"}')
  pp=$(awk -v t="$ttft" 'BEGIN{if(t>0)printf"%.0f",2048*1000.0/t;else print"NA"}')
  say "RESULT[$1]: decode_tps=$dec  prefill_tps=$pp  TTFT_ms=${ttft:-NA}  TPOT_ms=${tpot:-NA}  out_tps=${otps:-NA}  req_s=${reqs:-NA}"; }

say "=== WARMUP bench (discarded) ==="; bench 1 4 2048 128 >/dev/null 2>&1 || true
say "=== WARM benches ==="
report "c1.run1" "$(bench 1 6 2048 128)"
report "c1.run2" "$(bench 1 6 2048 128)"
report "c4.run1" "$(bench 4 16 2048 128)"

say "=== DONE. Stopping container. ==="; docker rm -f "$NAME" >/dev/null 2>&1
say "=== post-run health check ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"
say "stopped $NAME [$LABEL]"
