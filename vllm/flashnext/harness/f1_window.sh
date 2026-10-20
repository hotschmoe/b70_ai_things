#!/bin/bash
# F1 Flash-Next sidecar PREFILL window. BOTH cards.
# health before -> vllm serve (sidecar, O4 f4 decode recipe + X3 prefill
# switches + B70_EP_PREFILL) -> f1_probe.py over HTTP -> stop -> health after.
#
#   TAG=p1 bash vllm/flashnext/harness/f1_window.sh
#
# Env: TAG (required), F1_EXTRA ("K=V K=V" passed into the container),
# MAX_LEN (147456), MAX_BATCHED (12288), MAX_SEQS (2), PORT (18140),
# F1_LENGTHS, F1_COHERENCE_LEN, F1_REPEAT, F1_DECODE (probe knobs),
# SKIP_HEALTH_BEFORE=1, SERVE_TIMEOUT (seconds, default 5400).
set -uo pipefail

REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
TAG="${TAG:?TAG}"
OUT="$B/f1/run-$TAG"
mkdir -p "$OUT"
HARD_KB=8388608
NAME="vllm-fn-f1-$TAG"
PORT="${PORT:-18140}"
EP_SO="${F1_EP_SO:-$B/f1/b70_ep_C.f1.so}"
MOE_SO="${F1_MOE_SO:-$B/o4/_C.o4pin-o3g.so}"
WIDE_SO="${F1_WIDE_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.x3wide.so}"
HF=/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext
GGUF=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
ALIAS=qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-f1-prefill
exec > >(tee -a "$OUT/window.log") 2>&1

echo "=== F1 window $TAG $(date -u +%H:%M:%S) ==="
for f in "$EP_SO" "$MOE_SO" "$WIDE_SO" "$HF/config.json" "$GGUF"; do
  [ -f "$f" ] || { echo "missing $f"; echo WINDOW_DONE; exit 2; }
done
sha256sum "$EP_SO" "$MOE_SO" "$WIDE_SO" | cut -c1-16
AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "MEM_AVAILABLE_KB:$AVAIL_KB"
if [ "$AVAIL_KB" -lt 60000000 ]; then
  echo "ABORT MemAvailable under 60 GB before a Flash-Next load"; echo WINDOW_DONE; exit 4
fi

if [ "${SKIP_HEALTH_BEFORE:-0}" != 1 ]; then
  B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=28800 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-before.txt" 2>&1
  echo "HEALTH_BEFORE_RC:$?"; tail -n 2 "$OUT/health-before.txt"
  grep -q "xpu-health: HEALTHY" "$OUT/health-before.txt" || { echo "health before failed"; echo WINDOW_DONE; exit 3; }
fi

PYPATH="$B/c7b-fa-overlay:$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_ep_xpu:$REPO/vllm/flashnext/plugins/b70_qsa_xpu:$REPO/vllm/flashnext/harness"
EXTRA=()
for kv in ${F1_EXTRA:-}; do EXTRA+=(--env "$kv"); done

# RAM watchdog: docker-stop if MemAvailable < 8 GiB.
( while true; do
    a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$a" -lt "$HARD_KB" ]; then echo "ABORT_LOW_RAM $a $(date -u +%H:%M:%S)"; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 0; fi
    sleep 2
  done ) &
WD=$!

echo "=== serve $(date -u +%H:%M:%S) ==="
B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=28800 timeout --signal=TERM --kill-after=30 "${SERVE_TIMEOUT:-5400}" \
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card both --memory 100g --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 --env B70_EP=1 --env B70_QSA="${B70_QSA:-0}" \
  --env "B70_GGUF_PATH=$GGUF" --env "B70_GGUF_SO=$MOE_SO" --env "B70_EP_SO=$EP_SO" \
  --env "B70_EP_PROFILE=/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin" \
  --env "B70_EP_Q0=${B70_EP_Q0:-138}" --env B70_EP_SPLIT=profile \
  --env B70_PLE_NBYTES=28800138240 --env B70_EP_MEM_MIN_KB=$HARD_KB --env B70_EP_MAX_T=8192 \
  --env B70_XPU_CG_RECLAIM=500 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  --env B70_MTP_RT=/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt --env B70_PLE_MLOCK=1 --env B70_PLE_FAST=1 \
  --env B70_DECODE_HC=triton --env B70_DECODE_LEAN=1 --env B70_EP_SHARED_OVERLAP=1 --env B70_EP_FUSE_SHARED=1 \
  --env B70_EP_PREFILL=1 --env "B70_EP_PREFILL_MAX_T=${MAX_BATCHED:-12288}" --env "B70_EP_PREFILL_CTL=$OUT/ctl.txt" \
  --env B70_PREFILL=1 --env "B70_PREFILL_WIDE_SO=$WIDE_SO" --env B70_PREFILL_DQ_OVERLAP=0 \
  --env B70_PREFILL_REDUCE=grouped --env "B70_PREFILL_EXPERT_TILE=${B70_PREFILL_EXPERT_TILE:-64}" \
  --env "B70_PREFILL_BUCKET=${B70_PREFILL_BUCKET:-1}" --env "B70_PREFILL_DQ_T=${B70_PREFILL_DQ_T:-8192}" \
  --env B70_PREFILL_PLE_HASH=numpy --env B70_PREFILL_HC=triton \
  --env "PYTHONPATH=$PYPATH" --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 --env VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-f1" \
  "${EXTRA[@]}" \
  -- vllm serve "$HF" --host 0.0.0.0 --port "$PORT" \
  --served-model-name hotschmoe-dev "$ALIAS" \
  --tensor-parallel-size 1 --dtype bfloat16 --load-format b70_gguf --quantization b70_gguf \
  --language-model-only --trust-remote-code \
  --max-model-len "${MAX_LEN:-147456}" --max-num-seqs "${MAX_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_BATCHED:-12288}" \
  --gpu-memory-utilization "${GPU_MEM:-0.98}" --no-enable-prefix-caching \
  --mamba-cache-mode align --gdn-prefill-backend sycl --attention-backend FLASH_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,8]}' \
  > "$OUT/serve.log" 2>&1 &
SERVE=$!
touch "$OUT/.started"

# Wait for HTTP ready (load ~4 min + 12288-token profile + mlock).
READY=0
for i in $(seq 1 360); do
  if ! kill -0 $SERVE 2>/dev/null; then break; fi
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" 2>/dev/null && grep -q hotschmoe-dev "$OUT/models.json"; then READY=1; break; fi
  sleep 5
done
echo "READY:$READY $(date -u +%H:%M:%S)"
if [ "$READY" = 1 ]; then
  grep -E "EP-PF|b70_ep: prefill|Free memory|KV cache size|maximum concurrency|EP init|split layer 47" "$OUT/serve.log" | tail -n 12
  F1_BASE="http://127.0.0.1:$PORT" F1_OUT="$OUT/probe.json" F1_ALIAS="$ALIAS" F1_CTL="$OUT/ctl.txt" \
    "$REPO/bin/cpu-run" --quiet /usr/bin/python3 "$REPO/vllm/flashnext/harness/f1_probe.py" > "$OUT/probe.log" 2>&1
  echo "PROBE_RC:$? $(date -u +%H:%M:%S)"
  grep -E "^F1 " "$OUT/probe.log" | tail -n 30
  grep -E "EP-PF PROFILE|EP-PF flag|Traceback|Error" "$OUT/serve.log" | tail -n 20
else
  echo "serve not ready; tail:"; tail -n 40 "$OUT/serve.log"
fi
kill $WD 2>/dev/null
docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
wait $SERVE 2>/dev/null
echo "SERVE_RC:$? $(date -u +%H:%M:%S)"

B70_AGENT=F1 B70_GPU_LOCK_TIMEOUT=3600 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 2 "$OUT/health-after.txt"
echo WINDOW_DONE
