#!/usr/bin/env bash
# D1: bounded card-1 attention microprobe, no model load.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/d1/${D1_TAG:-fa-probe-1}"
mkdir -p "$OUT"
export B70_AGENT=D1 B70_GPU_LOCK_TIMEOUT=1800 B70_CPU_LOCK_TIMEOUT=1800
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1 || exit 3
"$R/bin/cpu-run" --quiet "$R/vllm/flashnext/harness/xpu-run.sh" \
    --card 1 --name vllm-fn-d1-fa --memory 8g \
    --env "PYTHONPATH=$B/c7b-fa-overlay:$B/vllm" \
    --env PYTHONUNBUFFERED=1 --env SYCL_CACHE_PERSISTENT=0 \
    -- /opt/venv/bin/python "$R/vllm/flashnext/harness/d1_fa_probe.py" \
    > "$OUT/probe.log" 2>&1
rc=$?
docker stop -t 10 vllm-fn-d1-fa >/dev/null 2>&1 || true
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
health=$?
printf 'D1_PROBE_RC:%s HEALTH_RC:%s\n' "$rc" "$health" | tee "$OUT/status.txt"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$health"
