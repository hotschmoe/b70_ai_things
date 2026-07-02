#!/usr/bin/env bash
# push_ar_ab.sh -- EXPERIMENT (2026-07-02): port the hand-rolled PUSH all-reduce (P2P_GPU.md J.7-K,
# proven on vLLM: decode AR ~34-45us vs oneCCL ~85-88us, prefill ~10.6 vs 9.4 GB/s) into the SGLANG
# W8A8 fused+MTP daily-driver config via sglang/patches/push_ar_xpu.py (chained from the repo
# woq_shim.py under B70_XPU_PUSH_AR=1; the woq_shim mount is REQUIRED -- the baked copy lacks the hook).
#
# Decode at TP=2 does ~128 x 10KB all-reduces per verify forward (latency-bound); push-AR halves the
# per-AR latency and is P2PACCESS-independent (no H.13 wedge surface). Expected: +10-30% decode t/s.
# CONTROL = the same-day shelf baseline (rdy_to_serve/sglang/qwen36-27b-w8a8 serve.sh run).
#
#   /mnt/vm_8tb/b70/gpu-run bash sglang/push_ar_ab.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_pushar
PORT=31003
CKPT=/models/qwen3.6-27b/w8a8-sqgptq
TOK=/models/qwen3.6-27b/bf16
SERVED=qwen36-27b-w8a8-mtp
KDIR=$ROOT/w8a8_kernel
PUSHDIR=$REPO/vllm/contrib/vllm_push_allreduce/prebuilt
CTX=8192
say(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ say "cleanup: rm $NAME"; docker rm -f "$NAME" >/dev/null 2>&1; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
trap cleanup EXIT

say "pre-flight xpu-health"
"$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; exit 3; }
docker rm -f "$NAME" >/dev/null 2>&1

say "launch W8A8 fused+MTP TP=2 + PUSH-AR -> :$PORT"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
  -v "$PUSHDIR:/work/push_ar:ro" \
  -v "$REPO/sglang/patches/woq_shim.py:/opt/venv/lib/python3.12/site-packages/woq_shim.py:ro" \
  -v "$REPO/sglang/patches/push_ar_xpu.py:/opt/venv/lib/python3.12/site-packages/push_ar_xpu.py:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  -e B70_XPU_PUSH_AR=1 -e PUSH_AR_SO=/work/push_ar/libxpu_push_ar_graph.so -e B70_PUSH_AR_STATS=1 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps 10 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 11 --speculative-draft-attention-backend triton --disable-cuda-graph \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --tp 2 --context-length $CTX --mem-fraction-static 0.90 --max-running-requests 4 --skip-server-warmup \
    --host 0.0.0.0 --port $PORT" >/dev/null

say "waiting for /health (load + spec JIT ~3-6min)..."
for i in $(seq 1 140); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1 | tail -50; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null || echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done

say "push-AR engagement check:"
docker logs "$NAME" 2>&1 | grep -E "push-ar" | head -10 || say "WARN: no push-ar log lines"

bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w8a8-mtp-PUSHAR"
rc=$?
say "final push-ar stats + engagement:"
docker logs "$NAME" 2>&1 | grep -E "push-ar" | tail -10

# --- best-effort torch profiler trace of the decode loop (launch-vs-AR-vs-compute decomposition) ---
if [ "${PROFILE:-1}" = 1 ]; then
  say "profiling ~400 decode steps (best-effort)"
  mkdir -p "$ROOT/sgl_cache/profile_pushar"
  curl -s -X POST "http://localhost:$PORT/start_profile" -H 'content-type: application/json' \
    -d '{"output_dir":"/sgl_cache/profile_pushar","num_steps":400,"activities":["CPU","GPU"]}' && echo
  curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"Explain how a B-tree differs from a hash index in a database engine, in detail.\",\"max_tokens\":512,\"temperature\":0,\"ignore_eos\":true}" >/dev/null
  curl -s -X POST "http://localhost:$PORT/stop_profile" && echo
  sleep 10
  say "profile artifacts:"; ls -la "$ROOT/sgl_cache/profile_pushar/" 2>/dev/null | tail -5
fi
exit $rc
