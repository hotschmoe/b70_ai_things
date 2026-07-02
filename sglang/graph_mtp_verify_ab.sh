#!/usr/bin/env bash
# graph_mtp_verify_ab.sh -- EXPERIMENT (2026-07-02, capture campaign Step 2): capture the MTP
# TARGET_VERIFY forward with torch.xpu.XPUGraph while the DRAFT chain stays EAGER.
#
# Why: the w8a8 TP=2 MTP daily driver decodes at ~25 t/s with a ~220ms fully-eager iteration
# (verify M=11 forward + 10 draft steps); all-reduces are only ~11ms of that (push-AR A/B: +0%).
# The iteration is LAUNCH/PYTHON-bound -> capturing the verify forward is the big lever.
#
# New pieces vs prod serve:
#   - xpu_cudagraph.py: TARGET_VERIFY (topk=1) static metadata hooks (this session, codex-drafted)
#     + draft-capture SKIP (B70_XPU_DRAFT_GRAPH!=1 -> draft eager; the 06-28 draft-graph replay HUNG)
#   - B70_XPU_CUDAGRAPH=1 + NO --disable-cuda-graph (fork default: decode backend 'full')
#   - mamba/GDN side: hybrid_linear_attn_backend already has target_verify out_graph support (fork)
#
# GATES (in order): (1) serve reaches /health with graphs captured ("cuda graph: True" in decode
# logs); (2) coherence; (3) perf_regime c1/c4/soak. Any hang -> the 140x5s wait times out, logs dumped.
#   /mnt/vm_8tb/b70/gpu-run bash sglang/graph_mtp_verify_ab.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_gmtpv
PORT=31004
CKPT=/models/qwen3.6-27b/w8a8-sqgptq
TOK=/models/qwen3.6-27b/bf16
SERVED=qwen36-27b-w8a8-mtp
KDIR=$ROOT/w8a8_kernel
CTX="${CTX:-8192}"
MAXBS="${MAXBS:-4}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ say "cleanup: rm $NAME"; docker rm -f "$NAME" >/dev/null 2>&1; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
trap cleanup EXIT

say "pre-flight xpu-health"
"$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; exit 3; }
docker rm -f "$NAME" >/dev/null 2>&1

say "launch W8A8 fused+MTP TP=2, TARGET_VERIFY captured / draft eager -> :$PORT"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/woq_shim.py:/opt/venv/lib/python3.12/site-packages/woq_shim.py:ro" \
  -v "$REPO/sglang/patches/xpu_cudagraph.py:/opt/venv/lib/python3.12/site-packages/xpu_cudagraph.py:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -v "$REPO/sglang/patches/mtp_tree_xpu.py:/opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py:ro" \
  -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  -e B70_XPU_CUDAGRAPH=1 -e B70_XPU_CUDAGRAPH_DEBUG=1 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps 10 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 11 --speculative-draft-attention-backend triton \
    --cuda-graph-max-bs $MAXBS \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --tp 2 --context-length $CTX --mem-fraction-static 0.87 --max-running-requests 4 --skip-server-warmup \
    --host 0.0.0.0 --port $PORT" >/dev/null

say "waiting for /health (load + spec JIT + graph capture, up to ~15min)..."
for i in $(seq 1 180); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1 | tail -60; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null || echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null || echo 000)" = 200 ] || { say "NEVER healthy -- logs:"; docker logs "$NAME" 2>&1 | tail -80; exit 1; }

say "capture evidence:"
docker logs "$NAME" 2>&1 | grep -E "xpu-cudagraph|Capture.*graph|cuda graph|draft cuda-graphs" | head -20

say "first gen (120s timeout; a HANG here = the captured verify path deadlock)"
g=$(curl -s --max-time 120 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":128,\"temperature\":0}")
echo "$g" | head -c 400; echo
echo "$g" | grep -q "content" || { say "FIRST GEN FAILED/HUNG -- logs:"; docker logs "$NAME" --since 3m 2>&1 | tail -60; exit 1; }

say "decode-log check (want: cuda graph: True on Decode batch lines)"
docker logs "$NAME" --since 3m 2>&1 | grep -E "Decode batch" | tail -3

bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w8a8-mtp-VERIFYGRAPH"
rc=$?
docker logs "$NAME" 2>&1 | grep -E "Decode batch" | tail -3
exit $rc
