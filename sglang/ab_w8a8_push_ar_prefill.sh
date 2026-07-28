#!/usr/bin/env bash
# Controlled large-prefill A/B for the sglang 0.5.6 W8A8 shelf stack.
#
# The 2026-07-02 eager push-AR experiment already closed tiny decode all-reduces
# at +0%. This test gates the custom transport to tensors >=64K elements so
# decode stays on oneCCL and asks the remaining question: do large EXTEND
# all-reduces improve cold prefill?
#
# Caller must hold both cards:
#   ./bin/gpu-run bash sglang/ab_w8a8_push_ar_prefill.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-sglang-xpu:mtp}"
PORT="${PORT:-30000}"
CTX="${CTX:-65536}"
MAXREQ="${MAXREQ:-4}"
MEMFRAC="${MEMFRAC:-0.90}"
MIN_NUMEL="${MIN_NUMEL:-65536}"
MAX_BYTES="${MAX_BYTES:-536870912}"
C1_LENS="${C1_LENS:-512,2048,8192,32768}"
C4_LENS="${C4_LENS:-2048,8192}"
C1_REPS="${C1_REPS:-3}"
C4_REPS="${C4_REPS:-2}"
RUN_COHERENCE="${RUN_COHERENCE:-1}"
RUN_DECODE="${RUN_DECODE:-1}"
NAME_OFF="${NAME_OFF:-sglang_w8a8_prefill_ar_off}"
NAME_ON="${NAME_ON:-sglang_w8a8_prefill_ar_on}"
CKPT="/models/qwen3.6-27b/w8a8-sqgptq"
KDIR="$ROOT/w8a8_kernel"
PUSHDIR="$REPO/vllm/contrib/vllm_push_allreduce/prebuilt"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  docker stop -t 30 "$NAME_OFF" "$NAME_ON" >/dev/null 2>&1 || true
  docker rm "$NAME_OFF" "$NAME_ON" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

start_arm() {
  local label="$1"
  local enabled="$2"
  local name="$3"
  local served="$4"

  echo "ARM -> $label push_ar=$enabled ctx=$CTX min_numel=$MIN_NUMEL max_bytes=$MAX_BYTES"
  "$REPO/bin/xpu-health"
  docker rm -f "$name" >/dev/null 2>&1 || true

  docker run -d --name "$name" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" \
    -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$KDIR:/work/kernel:ro" \
    -v "$PUSHDIR:/work/push_ar:ro" \
    -v "$REPO/sglang/patches/woq_shim.py:/opt/venv/lib/python3.12/site-packages/woq_shim.py:ro" \
    -v "$REPO/sglang/patches/push_ar_xpu.py:/opt/venv/lib/python3.12/site-packages/push_ar_xpu.py:ro" \
    -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
    -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
    -e HF_HOME=/hf_cache \
    -e XDG_CACHE_HOME=/sgl_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e CCL_TOPO_P2P_ACCESS=0 \
    -e B70_XPU_MTP=1 \
    -e B70_XPU_W8A8=1 \
    -e B70_XPU_W8A8_FUSED=1 \
    -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
    -e "B70_XPU_PUSH_AR=$enabled" \
    -e PUSH_AR_SO=/work/push_ar/libxpu_push_ar_graph.so \
    -e PUSH_AR_GRAPH=0 \
    -e "PUSH_AR_MIN_NUMEL=$MIN_NUMEL" \
    -e "PUSH_AR_MAXB=$MAX_BYTES" \
    -e B70_PUSH_AR_STATS=1 \
    -e PUSH_AR_STATS_EVERY=2000 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server \
      --model-path '$CKPT' \
      --served-model-name '$served' \
      --trust-remote-code \
      --device xpu \
      --attention-backend intel_xpu \
      --linear-attn-backend triton \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 10 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 11 \
      --speculative-draft-attention-backend triton \
      --disable-cuda-graph \
      --mamba-ssm-dtype float32 \
      --disable-overlap-schedule \
      --page-size 64 \
      --disable-radix-cache \
      --skip-server-warmup \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --tp 2 \
      --context-length '$CTX' \
      --mem-fraction-static '$MEMFRAC' \
      --max-running-requests '$MAXREQ' \
      --host 0.0.0.0 \
      --port '$PORT'" >/dev/null

  local healthy=0
  for _ in $(seq 1 140); do
    if ! docker ps --filter "name=^/${name}$" --format '{{.Names}}' | rg -qx "$name"; then
      echo "RESULT[$label] -> FAIL: container exited"
      docker logs "$name" 2>&1 | tail -100
      return 1
    fi
    if [ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 \
      "http://127.0.0.1:$PORT/health" 2>/dev/null || true)" = 200 ]; then
      healthy=1
      break
    fi
    sleep 5
  done
  [ "$healthy" = 1 ] || {
    echo "RESULT[$label] -> FAIL: health timeout"
    docker logs "$name" 2>&1 | tail -100
    return 1
  }

  local model_json model_id model_len
  model_json="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models")"
  model_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"$model_json")"
  model_len="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' <<<"$model_json")"
  echo "IDENTITY[$label] -> id=$model_id max_model_len=$model_len"
  [ "$model_id" = "$served" ] && [ "$model_len" -ge "$CTX" ]

  if [ "$enabled" = 1 ]; then
    local hits
    hits="$(docker logs "$name" 2>&1 | rg -c \
      'patched sglang XpuCommunicator.all_reduce' || true)"
    hits="${hits:-0}"
    echo "PATCH[$label] -> push_ar_patch_hits=$hits"
    [ "$hits" -ge 2 ] || {
      echo "RESULT[$label] -> FAIL: push-AR patch missing on one or both TP ranks"
      return 1
    }
  fi
}

bench_arm() {
  local label="$1"
  local enabled="$2"
  local name="$3"
  local served="$4"

  if [ "$RUN_COHERENCE" = 1 ]; then
    echo "COHERENCE[$label] -> 18 mixed prefill/decode streams"
    python3 "$REPO/vllm/gate_concurrent_coherence.py" \
      "http://127.0.0.1:$PORT/v1" "$served" 3 6 200
  fi

  echo "PREFILL[$label] -> c1 unique cold lens=$C1_LENS reps=$C1_REPS"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$served" 1 8 "$C1_LENS" "$C1_REPS"

  echo "PREFILL[$label] -> c4 unique cold lens=$C4_LENS reps=$C4_REPS"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$served" 4 8 "$C4_LENS" "$C4_REPS"

  if [ "$RUN_DECODE" = 1 ]; then
    echo "DECODE[$label] -> code c1"
    python3 "$REPO/vllm/nvfp4/bench_code.py" \
      "http://127.0.0.1:$PORT/v1" "$served" 1 256 2
  fi

  if [ "$enabled" = 1 ]; then
    local engaged_hits
    echo "ENGAGEMENT[$label] -> push-AR logs"
    docker logs "$name" 2>&1 | rg 'push-ar|argraph' | tail -30
    engaged_hits="$(docker logs "$name" 2>&1 | rg -c \
      '\[push-ar\] ENGAGED: sglang .* -> push collective' || true)"
    engaged_hits="${engaged_hits:-0}"
    echo "ENGAGEMENT[$label] -> engaged_hits=$engaged_hits"
    [ "$engaged_hits" -ge 1 ] || {
      echo "RESULT[$label] -> FAIL: push-AR never engaged"
      return 1
    }
  fi

  if docker logs "$name" 2>&1 | rg -i \
    'device_lost|out_of_resources|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' >/dev/null; then
    echo "RESULT[$label] -> FAIL: fatal or garbage marker in server log"
    return 1
  fi

  echo "RESULT[$label] -> PASS"
  docker stop -t 30 "$name" >/dev/null
  docker rm "$name" >/dev/null
  "$REPO/bin/xpu-health"
}

run_arm() {
  start_arm "$@"
  bench_arm "$@"
}

run_arm \
  "sglang-0.5.6-prefill-ar-off" \
  0 \
  "$NAME_OFF" \
  "qwen36-27b-w8a8-gptq-mtp-sgl056-prefill-ar-off"

run_arm \
  "sglang-0.5.6-prefill-ar-on" \
  1 \
  "$NAME_ON" \
  "qwen36-27b-w8a8-gptq-mtp-sgl056-prefill-ar-on"

echo "VERDICT -> both sglang W8A8 large-prefill push-AR A/B arms completed"
