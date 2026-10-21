#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
OUT=$B/x5/gdn-bf16/spec-r1
mkdir -p "$OUT"
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
set +e
"$REPO/vllm/flashnext/harness/xpu-run.sh" --card 1 --name vllm-fn-x5-gdn-spec \
  --env "B70_GDN_SO=$B/x5/gdn-bf16/libb70_gdn_xpu.so" --env "X5_PROBE_OUT=$OUT/numeric.json" \
  --env "PYTHONPATH=$B/vllm:$REPO/vllm/flashnext/harness" --env SYCL_CACHE_PERSISTENT=0 \
  -- /opt/venv/bin/python "$REPO/vllm/flashnext/harness/x5_gdn_spec_probe.py" > "$OUT/probe.log" 2>&1
RC=$?
set -e
"$REPO/bin/gpu-run" --card 1 "$REPO/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-after.txt"
echo "X5_GDN_SPEC_DONE rc=$RC"
exit "$RC"
