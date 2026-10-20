#!/bin/bash
# O4 Flash-Next sidecar decode profile window. BOTH cards.
# health before -> in-process profile (o4_decode_profile.py) -> health after.
#
#   TAG=base UT=0 bash vllm/flashnext/harness/o4_window.sh
#
# Env: TAG (required), UT=1 wraps the process in unitrace (kernel timing),
# O4_SO (MoE library; default the pinned copy of O3 _C.o3f.so), O4_EXTRA
# ("K=V K=V" passed into the container), O4_TOKENS, SKIP_HEALTH_BEFORE=1.
set -uo pipefail

REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
TAG="${TAG:?TAG}"
OUT="$B/o4/run-$TAG"
mkdir -p "$OUT"
HARD_KB=8388608
NAME="vllm-fn-o4-$TAG"
SO="${O4_SO:-$B/o4/_C.o4pin-o3f.so}"
UTBIN=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
exec > >(tee -a "$OUT/window.log") 2>&1

echo "=== O4 window $TAG $(date -u +%H:%M:%S) ==="
sha256sum "$SO" "$REPO/vllm/flashnext/plugins/b70_ep_xpu/b70_ep_xpu/_C.so" | cut -c1-16
AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "MEM_AVAILABLE_KB:$AVAIL_KB"
if [ "$AVAIL_KB" -lt 60000000 ]; then
  echo "ABORT MemAvailable under 60 GB before a Flash-Next load"; echo WINDOW_DONE; exit 4
fi

if [ "${SKIP_HEALTH_BEFORE:-0}" != 1 ]; then
  B70_AGENT=O4 B70_GPU_LOCK_TIMEOUT=14400 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-before.txt" 2>&1
  echo "HEALTH_BEFORE_RC:$?"; tail -n 3 "$OUT/health-before.txt"
  grep -q "xpu-health: HEALTHY" "$OUT/health-before.txt" || { echo "health before failed"; echo WINDOW_DONE; exit 3; }
fi

PYPATH="$B/c7b-fa-overlay:$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_ep_xpu:$REPO/vllm/flashnext/plugins/b70_qsa_xpu:$REPO/vllm/flashnext/harness"
EXTRA=()
for kv in ${O4_EXTRA:-}; do EXTRA+=(--env "$kv"); done

CMD=(/opt/venv/bin/python "$REPO/vllm/flashnext/harness/o4_decode_profile.py")
if [ "${UT:-0}" = 1 ]; then
  # The container lacks the L0 tracing layer; a host copy sits in o4/zelayers.
  CMD=(bash -c "export LD_LIBRARY_PATH=$B/o4/zelayers:\$LD_LIBRARY_PATH; exec $UTBIN -d --chrome-kernel-logging --teardown-on-signal 10 --session o4$TAG --follow-child-process 0 --output-dir-path $OUT/trace -o $OUT/trace/timing.txt ${CMD[*]}")
  mkdir -p "$OUT/trace"
  EXTRA+=(--env "O4_UT_SESSION=o4$TAG")
fi

# RAM watchdog: docker-stop if MemAvailable < 8 GiB.
( while true; do
    a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$a" -lt "$HARD_KB" ]; then echo "ABORT_LOW_RAM $a"; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 0; fi
    sleep 2
  done ) &
WD=$!

echo "=== run $(date -u +%H:%M:%S) ==="
touch "$OUT/.started"
B70_AGENT=O4 B70_GPU_LOCK_TIMEOUT=14400 timeout --signal=TERM --kill-after=30 30m \
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card both --memory 100g --name "$NAME" \
  --env B70_GGUF=1 --env B70_EP=1 --env B70_QSA=0 \
  --env "B70_GGUF_PATH=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
  --env "B70_GGUF_SO=$SO" \
  --env "B70_EP_PROFILE=/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin" \
  --env "B70_EP_Q0=${B70_EP_Q0:-138}" --env B70_EP_SPLIT=profile \
  --env B70_PLE_NBYTES=28800138240 --env B70_EP_MEM_MIN_KB=$HARD_KB --env B70_EP_MAX_T=8192 \
  --env B70_XPU_CG_RECLAIM=500 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  --env "PYTHONPATH=$PYPATH" --env PYTHONUNBUFFERED=1 --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 \
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-o4" \
  --env "O4_OUT=$OUT" --env "O4_TOKENS=${O4_TOKENS:-96}" \
  "${EXTRA[@]}" \
  -- "${CMD[@]}" > "$OUT/run.log" 2>&1
RC=$?
echo "RUN_RC:$RC $(date -u +%H:%M:%S)"
kill $WD 2>/dev/null
docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
grep -E "^O4 " "$OUT/run.log" | tail -n 30

B70_AGENT=O4 B70_GPU_LOCK_TIMEOUT=1800 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 3 "$OUT/health-after.txt"
echo WINDOW_DONE
