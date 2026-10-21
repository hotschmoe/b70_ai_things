#!/bin/bash
# O5 Flash-Next sidecar MTP parity+latency window. BOTH cards.
# health before -> in-process probe (o5_mtp_profile.py) -> health after.
#
#   TAG=m2 O5_MTP=2 bash vllm/flashnext/harness/o5_window.sh
#
# Env: TAG (required), O5_MTP (speculative tokens, 0 = off), UT=1 wraps the
# process in unitrace, O5_SO (MoE library; default O3's batch-invariant
# _C.o3g.so pinned copy), O5_EXTRA ("K=V K=V"; default = O4 f3 recipe +
# MTP sidecar), O5_TOKENS, O5_PROMPTS, O5_CAPTURE, SKIP_HEALTH_BEFORE=1.
set -uo pipefail

REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
TAG="${TAG:?TAG}"
OUT="$B/o5/run-$TAG"
mkdir -p "$OUT"
HARD_KB=8388608
NAME="vllm-fn-o5-$TAG"
SO="${O5_SO:-$B/o4/_C.o4pin-o3g.so}"
UTBIN=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
exec > >(tee -a "$OUT/window.log") 2>&1

echo "=== O5 window $TAG $(date -u +%H:%M:%S) ==="
sha256sum "$SO" "$REPO/vllm/flashnext/plugins/b70_ep_xpu/b70_ep_xpu/_C.so" | cut -c1-16
AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "MEM_AVAILABLE_KB:$AVAIL_KB"
if [ "$AVAIL_KB" -lt 60000000 ]; then
  echo "ABORT MemAvailable under 60 GB before a Flash-Next load"; echo WINDOW_DONE; exit 4
fi

if [ "${SKIP_HEALTH_BEFORE:-0}" != 1 ]; then
  B70_AGENT=O5 B70_GPU_LOCK_TIMEOUT=14400 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-before.txt" 2>&1
  echo "HEALTH_BEFORE_RC:$?"; tail -n 3 "$OUT/health-before.txt"
  grep -q "xpu-health: HEALTHY" "$OUT/health-before.txt" || { echo "health before failed"; echo WINDOW_DONE; exit 3; }
fi

PYPATH="${O5_PYPATH_PRE:+$O5_PYPATH_PRE:}$B/c7b-fa-overlay:$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_ep_xpu:$REPO/vllm/flashnext/plugins/b70_qsa_xpu:$REPO/vllm/flashnext/plugins/b70_o4_xpu:$REPO/vllm/flashnext/harness"
RECIPE="${O5_RECIPE:-$REPO/vllm/flashnext/harness/o5_recipe.env}"
cp "$RECIPE" "$OUT/recipe.env"
F3=$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep -v '^B70_GGUF_SO=' | xargs)
RSO=$(grep -v '^#' "$RECIPE" | sed 's/ *#.*//' | grep '^B70_GGUF_SO=' | cut -d= -f2)
[ -n "$RSO" ] && [ -z "${O5_SO:-}" ] && SO="$RSO"
echo "GGUF_SO: $SO $(sha256sum "$SO" | cut -c1-16)"
echo "O5_EXTRA: ${O5_EXTRA:-$F3} ${O5_EXTRA_ADD:-}"
EXTRA=()
for kv in ${O5_EXTRA:-$F3} ${O5_EXTRA_ADD:-}; do EXTRA+=(--env "$kv"); done

CMD=(/opt/venv/bin/python "$REPO/vllm/flashnext/harness/${O5_SCRIPT:-o5_mtp_profile.py}")
if [ "${UT:-0}" = 1 ]; then
  # The container lacks the L0 tracing layer; a host copy sits in o4/zelayers.
  CMD=(bash -c "export LD_LIBRARY_PATH=$B/o4/zelayers:\$LD_LIBRARY_PATH; exec $UTBIN -d --chrome-kernel-logging --teardown-on-signal 10 --session o5$TAG --follow-child-process 0 --output-dir-path $OUT/trace -o $OUT/trace/timing.txt ${CMD[*]}")
  mkdir -p "$OUT/trace"
  EXTRA+=(--env "O5_UT_SESSION=o5$TAG")
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
B70_AGENT=O5 B70_GPU_LOCK_TIMEOUT=14400 timeout --signal=TERM --kill-after=30 30m \
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
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-o5" \
  --env "O5_OUT=$OUT" --env "O5_TAG=$TAG" --env "O5_MTP=${O5_MTP:-0}" --env "O5_TOKENS=${O5_TOKENS:-128}" --env "O5_PROMPTS=${O5_PROMPTS:-4}" --env "O5_CAPTURE=${O5_CAPTURE:-}" --env "O5_SPEC_SCHEDULE=${O5_SPEC_SCHEDULE:-}" --env "O5_PROBE_TOKENS=${O5_PROBE_TOKENS:-4}" --env "O5_PROBE_PROMPT=${O5_PROBE_PROMPT:-Write a short story about a lighthouse keeper who discovers a message in a bottle.}" \
  "${EXTRA[@]}" \
  -- "${CMD[@]}" > "$OUT/run.log" 2>&1
RC=$?
echo "RUN_RC:$RC $(date -u +%H:%M:%S)"
kill $WD 2>/dev/null
docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
grep -E "^O5 " "$OUT/run.log" | tail -n 30

B70_AGENT=O5 B70_GPU_LOCK_TIMEOUT=1800 "$REPO/bin/gpu-run" "$REPO/bin/xpu-health" > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 3 "$OUT/health-after.txt"
echo WINDOW_DONE
