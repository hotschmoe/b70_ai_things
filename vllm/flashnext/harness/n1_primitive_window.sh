#!/usr/bin/env bash
# Schedule explicitly with the orchestrator. Physical card1, <=2min body.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/n1/run-${TAG:?TAG}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window.log") 2>&1
export B70_AGENT=N1 B70_GPU_LOCK_TIMEOUT=600
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
PRE=$?
if [ "$PRE" -ne 0 ] || ! rg -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"; then
  echo "PRE_HEALTH_FAIL:$PRE"; exit 3
fi
sha256sum "$B/n1/_C.so" "$REPO/vllm/flashnext/plugins/b70_o4_xpu/b70_o4_xpu/_C.so" > "$OUT/identity.sha256"
PYPATH="$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_o4_xpu:$REPO/vllm/flashnext/plugins/b70_n1_xpu"
# cpu-run outside the GPU lease. Acquire only after the parent grants a
# free-card gap, so a quiet lock is never held behind an active serve.
"$REPO/bin/cpu-run" --quiet timeout --signal=TERM --kill-after=15 150 \
  "$REPO/vllm/flashnext/harness/xpu-run.sh" --card 1 --memory 8g --name "vllm-fn-n1-${TAG}" \
  --env "PYTHONPATH=$PYPATH" --env "N1_OUT=$OUT" --env "B70_N1_SO=$B/n1/_C.so" \
  --env SYCL_CACHE_PERSISTENT=0 --env PYTHONUNBUFFERED=1 \
  -- /opt/venv/bin/python "$REPO/vllm/flashnext/harness/n1_primitive.py" > "$OUT/primitive.log" 2>&1
RC=$?
echo "N1_PRIMITIVE_RC:$RC"
tail -n 5 "$OUT/primitive.log"
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
POST=$?
echo "N1_POST_HEALTH_RC:$POST"
tail -n 3 "$OUT/health-after.txt"
[ "$RC" -eq 0 ] && [ "$POST" -eq 0 ] && rg -q 'N1_PRIMITIVE_OK' "$OUT/primitive.log" && rg -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
