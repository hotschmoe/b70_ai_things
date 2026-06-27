#!/usr/bin/env bash
# 122_w8a8_tp2_sglang_bench.sh -- the W8A8 TP=2 end-to-end perf test on sglang-XPU.
# This is the ONE live lever from the vLLM "fake 63 t/s" headline (W8A8 TP=2 MTP PIECEWISE):
# on sglang MTP is upstream-blocked (out_cache_loc) and XPU cudagraph degrades, so W8A8 TP=2
# ALONE is what we can actually measure -- and now coherently (sglang fixes the GDN NaN).
#
# Hypothesis (from the single-layer microbench, JOURNAL 2026-06-27):
#   - decode (M=1) ~launch-bound -> ~bf16 (~0.79x microbench; eager ceiling ~9 t/s)
#   - prefill 1.24-1.61x bf16 -> TTFT/PP win over bf16 TP=2's 3098 prefill_tps
#   - better accuracy than int4
# Compare vs scoreboard: bf16 TP=2 c1 decode 9.03 / prefill 3098 / TTFT 661; woq int4 TP=1 c1 9.44.
#
# Holds BOTH cards -> run under: ./bin/gpu-run bash scripts/122_w8a8_tp2_sglang_bench.sh
set -uo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="sglang-xpu:woq"
NAME="sglang_w8a8"
CKPT="/models/Qwen3.6-27B-W8A8-sqgptq"
SERVED="qwen36-27b-w8a8-sqgptq"
PORT=30000
TP=2
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
TOK="/models/Qwen_Qwen3.6-27B"
LOG="$REPO/sglang/w8a8_tp2_bench.log"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"

say "=== W8A8 TP=2 sglang serve: $SERVED  img=$IMG ctx=$CTX memfrac=$MEMFRAC ==="
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
    --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

# --- wait for health (watch for crash) ---
say "waiting for /health (model load ~3-5min)..."
ok=0
for i in $(seq 1 120); do  # up to ~600s
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    say "CONTAINER EXITED during load -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = 200 ]; then ok=1; say "/health 200 after ~$((i*5))s"; break; fi
  if docker logs "$NAME" 2>&1 | grep -qiE "coredumps before exiting|Scheduler hit an exception|Received sigquit"; then
    say "WORKER CRASHED (up but dead) -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break
  fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "SERVE DID NOT COME HEALTHY (ok=$ok). Tailing logs:"; docker logs "$NAME" 2>&1 | tail -60 | tee -a "$LOG"
  say "leaving container for inspection; exiting."; exit 1
fi

# --- coherence (must NOT be garbage / '!!!!') ---
say "=== coherence check ==="
gen=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Answer in two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
echo "$gen" | python3 -c "import sys,json; d=json.load(sys.stdin); print('COHERENCE:', repr(d['choices'][0]['message']['content'][:240]))" | tee -a "$LOG" || { say "coherence parse failed: $gen"; }

# --- warm bench: discard run 1, record runs 2-3 (B70 idle-downclock lesson) ---
bench(){ # conc num
  docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
    --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
    --random-input-len ${3:-2048} --random-output-len ${4:-128} --num-prompts $2 --max-concurrency $1 2>&1"
}
report(){ # label raw
  local raw="$2"
  local ttft tpot otps reqs dec pp
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  tpot=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  otps=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  reqs=$(echo "$raw"|grep -i 'Request throughput'|grep -oE '[0-9.]+'|head -1)
  dec=$(awk -v t="$tpot" 'BEGIN{if(t>0)printf"%.2f",1000.0/t;else print"NA"}')
  pp=$(awk -v t="$ttft" 'BEGIN{if(t>0)printf"%.0f",2048*1000.0/t;else print"NA"}')
  say "RESULT[$1]: decode_tps=$dec  prefill_tps=$pp  TTFT_ms=${ttft:-NA}  TPOT_ms=${tpot:-NA}  out_tps=${otps:-NA}  req_s=${reqs:-NA}"
}

say "=== WARMUP bench (discarded) ==="
bench 1 4 2048 128 >/dev/null 2>&1 || true
say "=== WARM benches ==="
report "c1.run1" "$(bench 1 6 2048 128)"
report "c1.run2" "$(bench 1 6 2048 128)"
report "c4.run1" "$(bench 4 16 2048 128)"

say "=== DONE. Stopping container to release lease. ==="
docker rm -f "$NAME" >/dev/null 2>&1
say "stopped $NAME"
