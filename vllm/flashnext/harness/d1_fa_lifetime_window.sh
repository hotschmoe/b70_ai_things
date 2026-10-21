#!/usr/bin/env bash
# Scratch-only; parent authorizes physicalcard1 <=3x60sbody +health.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/d1/${D1_TAG:?D1_TAG}"
[ ! -e "$OUT" ] || { echo "ABORT output exists: $OUT"; exit 2; }
mkdir -p "$OUT"
export B70_AGENT=D1 B70_GPU_LOCK_TIMEOUT=1800
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1 || exit 3
"$R/vllm/flashnext/harness/xpu-run.sh" --card 1 --memory 8g \
 --name vllm-fn-d1-lifetime \
 --env "PYTHONPATH=$B/d1/fa-overlay:$B/vllm" --env B70_XPU_CG_RECLAIM=500 \
 --env B70_RECLAIM_TRACE=1 --env "B70_STEP_LOG=$OUT/reclaim.log" \
 --env PYTHONUNBUFFERED=1 --env SYCL_CACHE_PERSISTENT=0 \
 -- bash -c 'for arm in original temporary persistent; do /opt/venv/bin/python /mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/d1_fa_lifetime_stress.py --arm "$arm" || exit $?; done' \
 > "$OUT/probe.log" 2>&1
rc=$?
docker stop -t 10 vllm-fn-d1-lifetime >/dev/null 2>&1 || true
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
health=$?
printf 'D1_LIFETIME_RC:%s HEALTH_RC:%s\n' "$rc" "$health" | tee "$OUT/status.txt"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$health"
