#!/usr/bin/env bash
# 125_w8a8_accuracy.sh -- HumanEval+ accuracy gate for the FUSED W8A8 kernels.
# Served via the fast MTP config (steps=10) -- MTP is greedy-LOSSLESS (accepts only target-argmax tokens)
# so its output == eager greedy output == the fused-kernel greedy decoding; same accuracy, ~3x faster eval.
# Validates the built int8_gemm ops don't degrade vs the base sqgptq W8A8 quant (ref: W4A8 same-stack 0.92/0.90).
# Serves detached + held (heartbeat to survive background), runs run_evals tier1, tears down.
set -uo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="/mnt/vm_8tb/github/b70_ai_things"
IMG="${IMG:-sglang-xpu:mtp}"; NAME="${NAME:-sglang_w8a8_acc}"; PORT=30000; TP=2
CKPT="${CKPT:-/models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp}"
SERVED="${SERVED:-qwen36-27b-w8a8-vision-mtp}"; QUANT="${QUANT:-w8a8-fused-vision}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"; SPEC_STEPS="${SPEC_STEPS:-10}"; SPEC_DRAFT="${SPEC_DRAFT:-11}"
MAXREQ="${MAXREQ:-4}"; CTX="${CTX:-8192}"; MEMFRAC="${MEMFRAC:-0.90}"; LIMIT="${LIMIT:-}"
LOG="$REPO/w8a8/w8a8_accuracy.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"

say "=== pre-flight xpu-health ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG" || { say "UNHEALTHY -- abort"; exit 3; }
say "=== serve W8A8 fused+MTP (steps=$SPEC_STEPS) for accuracy gate ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/models_w8a8:/models_w8a8:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --disable-cuda-graph \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache --skip-server-warmup \
    --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC --max-running-requests $MAXREQ \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (heartbeat to survive background)..."
ok=0
for i in $(seq 1 140); do
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    say "CONTAINER EXITED"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$LOG"; ok=2; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo 000)
  [ "$code" = 200 ] && { ok=1; say "/health 200 after ~$((i*5))s"; break; }
  [ $((i % 6)) -eq 0 ] && say "  ...loading (${i}x5s, http=$code)"   # heartbeat
  sleep 5
done
[ "$ok" = 1 ] || { say "NOT healthy"; docker rm -f "$NAME">/dev/null 2>&1; "$REPO/bin/xpu-health" 2>&1|tail -2|tee -a "$LOG"; exit 1; }

say "=== coherence ==="
curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":64,\"temperature\":0}" \
  | python3 -c "import sys,json;print('COHERENCE:',repr(json.load(sys.stdin)['choices'][0]['message']['content'][:160]))" 2>&1 | tee -a "$LOG" || say "coherence parse fail"

say "=== run_evals tier1 HumanEval+ (sandboxed) ${LIMIT:+limit=$LIMIT} ==="
"$REPO/evals/.venv/bin/python" "$REPO/evals/orchestrator/run_evals.py" \
  --endpoint "http://localhost:$PORT/v1" --model "$SERVED" --quant "$QUANT" \
  --tiers 1 --tier1-dataset humaneval --allow-code-exec --max-tokens 2048 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a "$LOG"

say "=== stopping serve ==="; docker rm -f "$NAME" >/dev/null 2>&1
say "=== post health ==="; "$REPO/bin/xpu-health" 2>&1 | tail -3 | tee -a "$LOG"
say "=== latest result dir ==="; ls -dt "$REPO"/evals/results/*"$QUANT"* 2>/dev/null | head -1 | tee -a "$LOG"
