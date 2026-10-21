#!/usr/bin/env bash
# Invoke only after the orchestrator explicitly grants physical card1.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/p1/run-${TAG:?TAG}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window.log") 2>&1
export B70_AGENT=P1 B70_GPU_LOCK_TIMEOUT=600
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
PRE=$?
if [ "$PRE" -ne 0 ] || ! rg -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"; then
  echo "P1_PRE_HEALTH_FAIL:$PRE"; exit 3
fi
sha256sum "$B/p1/_C.p1selected.so" "$B/f1/_C.f1bucket2.so" "$R/vllm/flashnext/plugins/b70_p1_xpu/csrc/p1_dequant.cpp" "$R/vllm/flashnext/plugins/b70_p1_xpu/csrc/p1_dequant.hpp" "$R/vllm/flashnext/plugins/b70_p1_xpu/csrc/p1_selected_register.cpp" "$R/vllm/flashnext/harness/p1_primitive.py" > "$OUT/identity-before.sha256"
PYPATH="$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py"
"$R/bin/cpu-run" --quiet timeout --signal=TERM --kill-after=15 210 \
  "$R/vllm/flashnext/harness/xpu-run.sh" --card 1 --memory 12g --name "vllm-fn-p1-${TAG}" \
  --env "PYTHONPATH=$PYPATH" --env SYCL_CACHE_PERSISTENT=0 --env PYTHONUNBUFFERED=1 \
  -- /opt/venv/bin/python "$R/vllm/flashnext/harness/p1_primitive.py" --out "$OUT/components.json" > "$OUT/components.log" 2>&1
RC=$?
echo "P1_BODY_RC:$RC"
tail -n 6 "$OUT/components.log"
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
POST=$?
echo "P1_POST_HEALTH_RC:$POST"
tail -n 3 "$OUT/health-after.txt"
sha256sum -c "$OUT/identity-before.sha256" > "$OUT/identity-after.txt"
ID=$?
[ "$RC" -eq 0 ] && [ "$POST" -eq 0 ] && [ "$ID" -eq 0 ] && rg -q 'P1_COMPONENT_OK' "$OUT/components.log" && rg -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
