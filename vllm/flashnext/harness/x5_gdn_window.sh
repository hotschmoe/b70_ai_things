#!/usr/bin/env bash
# Bounded card-1 BF16 GDN correctness probe; quiet only after lease looks free.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
OUT=$B/x5/gdn-bf16
mkdir -p "$OUT/${PROBE_TAG:-r2}"
RUN_OUT=$OUT/${PROBE_TAG:-r2}
NAME=vllm-fn-x5-gdn-probe
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$RUN_OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$RUN_OUT/health-before.txt"
set +e
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card 1 --name "$NAME" \
  --env "B70_GDN_SO=$OUT/libb70_gdn_xpu.so" --env "X5_PROBE_OUT=$RUN_OUT/numeric.json" \
  --env "PYTHONPATH=$B/vllm:$REPO/vllm/flashnext/harness" --env SYCL_CACHE_PERSISTENT=0 \
  -- /opt/venv/bin/python "$REPO/vllm/flashnext/harness/x5_gdn_probe.py" > "$RUN_OUT/probe.log" 2>&1
RC=$?
set -e
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$RUN_OUT/health-after.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$RUN_OUT/health-after.txt"
if [[ "$RC" == 0 ]]; then cp "$RUN_OUT/numeric.json" "$OUT/numeric.json"; fi
echo "X5_GDN_PROBE_DONE rc=$RC"
exit "$RC"
