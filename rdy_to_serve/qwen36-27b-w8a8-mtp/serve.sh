#!/usr/bin/env bash
# qwen36-27b-w8a8-mtp -- W8A8 (int8) + NEXTN MTP, the int8 all-rounder that HANDILY beats bf16/fp8 on
# prefill, TTFT, AND decode. Built fused int8 oneDNN ops (int8_gemm_w8a16 decode fp16-act /
# int8_gemm_w8a8 prefill s8-act) + NEXTN chain-MTP (steps=10) on the grafted W8A8 vision+MTP checkpoint.
#
#   c1 (IN2048/OUT128, warm, TP=2): DECODE ~25.2 t/s | PP ~4344 tok/s | TTFT ~471 ms
#   vs bf16 TP=2 (9.03 / 3098 / 661): TG +180% (2.8x), PP +40%, TTFT -29%.  > int4+MTP decode (15.3).
#   ACCURACY: HumanEval+ 0.970 / 0.933 (base/plus) -- HIGHER than int4 same-stack (0.933/0.896).
#   VISION retained (grafted vision+MTP ckpt). GREEDY-only (XPU MTP ignores sampling, like all XPU NEXTN).
#   Full head-to-head + build: ../../w8a8/W8A8_SGLANG_PLAN.md + ../../w8a8/W8A8_BUILD.md.
#
# RUNTIME MOUNTS (not a baked image):
#   - IMG = sglang-xpu:mtp  (baked XPU NEXTN gates + compressed_tensors W8A8 scheme + woqgemm)
#   - the built _xpu_C.abi3.so (B70_XPU_C_SO); B70_XPU_W8A8_FUSED=1 routes int8 linears to the fused ops
#   - the updated w8a8_shim.py (FUSED hybrid) over the baked copy
#   - the grafted ckpt at $ROOT/models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp (graft_mtp.py: vision+MTP head;
#     symlinks into /models, so BOTH /models and /models_w8a8 are mounted)
#   - in-container: source oneAPI setvars + PREPEND the oneAPI compiler lib to LD_LIBRARY_PATH (or the
#     ctypes-loaded .so resolves but torch loses the XPU device -- see ../../w8a8/W8A8_BUILD.md)
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh start   # serve TP=2 (both cards), coherence-gated, stay up
#   bash serve.sh stop                            # stop + release
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh run     # start + bench + stop in one lease
#
# [!] TP=2 = BOTH cards. cudagraph DISABLED (--disable-cuda-graph): W8A8 TP=2+MTP is stable that way; capture
#     is a CEILING at TP=2 (decode is all-reduce-bound, not launch-bound). For pure prefill/sampling use the
#     eager sibling (../../scripts/123_w8a8_fused_ab.sh, no MTP -> PP 4570 / TTFT 448 / decode 8.1, samples).
# [!] --max-running-requests 4 (spec mamba cache cap). >4 OOMs KV at ctx 8192; extra requests queue + complete.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_w8a8_mtp}"
CKPT="${CKPT:-/models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp}"
TOK="${TOK:-/models/Qwen_Qwen3.6-27B}"
SERVED="${SERVED:-qwen36-27b-w8a8-mtp}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
SPEC_STEPS="${SPEC_STEPS:-10}"; SPEC_DRAFT="${SPEC_DRAFT:-11}"; MAXREQ="${MAXREQ:-4}"
PORT="${PORT:-30000}"; TP=2; CTX="${CTX:-8192}"; MEMFRAC="${MEMFRAC:-0.90}"
LOG="${LOG:-$SCRIPT_DIR/serve.log}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start(){
  say "pre-flight xpu-health"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; return 3; }
  docker rm -f "$NAME" >/dev/null 2>&1
  say "serve W8A8 fused+MTP (steps=$SPEC_STEPS) TP=2 -> $SERVED on :$PORT (img=$IMG)"
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
      --host 0.0.0.0 --port $PORT" >/dev/null
  say "waiting for /health (load + spec JIT ~3-6min)..."
  for i in $(seq 1 140); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  local g; g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":64,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json;c=json.load(sys.stdin)['choices'][0]['message']['content'] or ''
print('COHERENCE OK:',repr(c[:160])) if c.strip() and (len(c)<16 or max(c.count(x) for x in set(c))/len(c)<0.6) else (print('GATE FAIL:',repr(c[:120])) or sys.exit(1))" \
    || { say "coherence gate FAILED -- see: docker logs $NAME"; return 1; }
  say "healthy + coherent; serving $SERVED on :$PORT"
}
stop(){ docker rm -f "$NAME" >/dev/null 2>&1; say "stopped $NAME"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
bench(){ docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
  python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT --served-model-name '$SERVED' \
  --tokenizer '$TOK' --dataset-name random --random-input-len 2048 --random-output-len 128 --num-prompts 6 --max-concurrency 1 2>&1" \
  | grep -iE 'Mean TTFT|Mean TPOT|Output token throughput'; }

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  bench) bench ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-Why is the sky blue?}\"}],\"max_tokens\":128,\"temperature\":0}" ;;
  run)   start && bench; rc=$?; stop; exit $rc ;;
  *) echo "usage: serve.sh {start|stop|bench|gen|run}"; exit 2 ;;
esac
