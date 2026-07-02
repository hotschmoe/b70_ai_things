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
SPEC_STEPS="${SPEC_STEPS:-10}"; SPEC_DRAFT="${SPEC_DRAFT:-$((SPEC_STEPS+1))}"
DRAFT_GRAPH="${DRAFT_GRAPH:-0}"   # 1 = also capture the draft chain (B70_XPU_DRAFT_GRAPH)
# The intel_xpu XMX mha kernel uses sycl work_group_scratch_memory -> NOT SYCL-Graph-recordable
# ("feature not yet available with SYCL Graph", first run 18:05). Split backends: XMX for eager
# PREFILL, triton (static SLM, graph-proven since int4-graph) for the captured DECODE/VERIFY path.
PREFILL_ATTN="${PREFILL_ATTN:-intel_xpu}"
DECODE_ATTN="${DECODE_ATTN:-triton}"
# GRAPH_BACKEND: full = one graph per bs (oneCCL records -> RUN-4 replay deadlock; needs push-AR or
# capture-safe collectives). breakable = segmented capture, attention/mamba AND (via xpu_cudagraph
# section 4) TP collectives run EAGER between segments -- the vLLM-PIECEWISE-equivalent shape.
# Under breakable the intel_xpu XMX attn is fine EVERYWHERE (runs eager), so ATTN_ARGS goes single-backend.
GRAPH_BACKEND="${GRAPH_BACKEND:-breakable}"
if [ "$GRAPH_BACKEND" = breakable ]; then
  ATTN_ARGS="--attention-backend intel_xpu"
else
  ATTN_ARGS="--prefill-attention-backend $PREFILL_ATTN --decode-attention-backend $DECODE_ATTN --speculative-attention-mode decode"
fi
# RUN 4 hang root-cause hypothesis: oneCCL all-reduces RECORDED into the captured verify graph deadlock
# at replay (host-staged half never re-executes) -> both ranks wedge, next eager D2H spins in
# appendUSMMemcpy (watchdog stack: mtp_tree_xpu _build_tree sl.tolist). Fix = the K.6 capturable
# push-AR: during capture, ARs record as device-side L0-event-synced posted writes (ar_allreduce_graph),
# proven coherent inside captured decode on vLLM (P2P_GPU.md K). PUSHAR=1 default.
PUSHAR="${PUSHAR:-0}"
PUSHDIR=$REPO/vllm/contrib/vllm_push_allreduce/prebuilt
say(){ echo "[$(date +%H:%M:%S)] $*"; }
LOGSAVE="$REPO/sglang/graph_mtp/last_run_$(date +%H%M%S).log"
cleanup(){ say "cleanup: rm $NAME (logs -> $LOGSAVE)"; docker logs "$NAME" >"$LOGSAVE" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
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
  -v "$PUSHDIR:/work/push_ar:ro" \
  -v "$REPO/sglang/patches/push_ar_xpu.py:/opt/venv/lib/python3.12/site-packages/push_ar_xpu.py:ro" \
  -e B70_XPU_PUSH_AR=$PUSHAR -e PUSH_AR_SO=/work/push_ar/libxpu_push_ar_graph.so -e PUSH_AR_GRAPH=$PUSHAR \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  -e B70_XPU_CUDAGRAPH=1 -e B70_XPU_CUDAGRAPH_DEBUG=1 -e B70_XPU_DRAFT_GRAPH=$DRAFT_GRAPH \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu $ATTN_ARGS --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton \
    --cuda-graph-max-bs $MAXBS --cuda-graph-backend-decode $GRAPH_BACKEND \
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

# the first heavy gen can stall the scheduler heartbeat -> transient /health 503; wait it out
say "re-confirming /health before bench (up to 3min)..."
for i in $(seq 1 36); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null || echo 000)" = 200 ] && break
  sleep 5
done

bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w8a8-mtp-VERIFYGRAPH"
rc=$?
docker logs "$NAME" 2>&1 | grep -E "Decode batch" | tail -3
exit $rc
