#!/usr/bin/env bash
# 124_w8a8_mtp.sh -- W8A8 FUSED + NEXTN MTP spec-decode, TP=2 (the decode-beats-bf16 lever).
# Combines the fused int8 hybrid (scripts/123 / w8a8_shim FUSED) with NEXTN chain-MTP (the int4 MTP recipe,
# rdy_to_serve/qwen36-27b-int4-mtp). MTP amortizes the TP=2 all-reduce/per-step tax across accepted tokens
# -> targets decode 8.3 -> ~12-13 t/s (handily beats bf16 9.0), keeping the prefill/TTFT win. Vision retained
# (grafted vision-mtp ckpt). cudagraph DISABLED (W8A8 TP=2+MTP stable that way). GREEDY-only on XPU.
#
# Holds BOTH cards -> run under: ./bin/gpu-run bash scripts/124_w8a8_mtp.sh
set -uo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="${IMG:-sglang-xpu:mtp}"          # baked XPU MTP gates (mtp_tree_xpu.py) + woqgemm + compressed_tensors
NAME="${NAME:-sglang_w8a8_mtp}"
CKPT="${CKPT:-/models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp}"  # grafted: vision(333)+W8A8 LM+MTP head
SERVED="${SERVED:-qwen36-27b-w8a8-vision-mtp}"
FUSED="${FUSED:-1}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
SPEC_STEPS="${SPEC_STEPS:-10}"   # W8A8 decode peak = 10 (int8-XMX verify is cheap -> deeper drafts win;
SPEC_DRAFT="${SPEC_DRAFT:-11}"   #   7->23.8, 10->25.25 (+6%), 12->24.35 drops. int4 peaked at 7.)
MAXREQ="${MAXREQ:-4}"                  # spec mamba cache cap
PORT=30000; TP=2
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
COHERENCE_ONLY="${COHERENCE_ONLY:-0}"
TOK="/models/Qwen_Qwen3.6-27B"
LOG="$REPO/w8a8/w8a8_mtp_steps${SPEC_STEPS}.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"

say "=== pre-flight xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG" || { say "UNHEALTHY pre-serve -- aborting"; exit 3; }

say "=== W8A8 FUSED + MTP TP=2: $SERVED steps=$SPEC_STEPS maxreq=$MAXREQ img=$IMG ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/models_w8a8:/models_w8a8:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  $( [ "$FUSED" = 1 ] && echo "-e B70_XPU_W8A8_FUSED=1" ) \
  ${DENV:+$(for kv in $DENV; do echo -n "-e $kv "; done)} \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --disable-cuda-graph \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache --skip-server-warmup \
    --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC --max-running-requests $MAXREQ \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (model load + spec JIT ~3-6min)..."
ok=0
for i in $(seq 1 140); do
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    say "CONTAINER EXITED -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = 200 ]; then ok=1; say "/health 200 after ~$((i*5))s"; break; fi
  if docker logs "$NAME" 2>&1 | grep -qiE "coredumps before exiting|Scheduler hit an exception|Received sigquit|DEVICE_LOST"; then
    say "WORKER CRASHED -- last 40 log lines:"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$LOG"; ok=2; break; fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "SERVE NOT HEALTHY (ok=$ok)."; docker rm -f "$NAME" >/dev/null 2>&1
  say "=== post health ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"; exit 1; fi

say "=== shim + MTP wiring ==="; docker logs "$NAME" 2>&1 | grep -iE "w8a8-fused|w8a8-shim|nextn|mtp|spec" | tail -8 | tee -a "$LOG"
say "=== coherence (greedy; must NOT be '!!!!') ==="
gen=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":80,\"temperature\":0}")
echo "$gen" | python3 -c "import sys,json; d=json.load(sys.stdin); print('COHERENCE:', repr(d['choices'][0]['message']['content'][:240]))" | tee -a "$LOG" || say "coherence parse failed: ${gen:0:200}"

if [ "$COHERENCE_ONLY" = 1 ]; then
  say "=== COHERENCE_ONLY: stopping. ==="; docker rm -f "$NAME" >/dev/null 2>&1
  say "=== post health ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"; exit 0; fi

bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT \
  --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
  --random-input-len ${3:-2048} --random-output-len ${4:-128} --num-prompts $2 --max-concurrency $1 2>&1"; }
report(){ local raw="$2" ttft tpot otps dec pp
  ttft=$(echo "$raw"|grep -i 'Mean TTFT'|grep -oE '[0-9.]+'|head -1)
  tpot=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  otps=$(echo "$raw"|grep -i 'Output token throughput'|grep -oE '[0-9.]+'|head -1)
  dec=$(awk -v t="$tpot" 'BEGIN{if(t>0)printf"%.2f",1000.0/t;else print"NA"}')
  pp=$(awk -v t="$ttft" 'BEGIN{if(t>0)printf"%.0f",2048*1000.0/t;else print"NA"}')
  say "RESULT[$1]: decode_tps=$dec prefill_tps=$pp TTFT_ms=${ttft:-NA} TPOT_ms=${tpot:-NA} out_tps=${otps:-NA}"; }
say "=== WARMUP (discard; JITs spec path) ==="; bench 1 3 2048 128 >/dev/null 2>&1 || true
say "=== WARM c1 ==="; report c1.run1 "$(bench 1 6 2048 128)"; report c1.run2 "$(bench 1 6 2048 128)"
say "=== DONE. stopping. ==="; docker rm -f "$NAME" >/dev/null 2>&1
say "=== post health ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"
say "stopped $NAME"
