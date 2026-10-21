#!/usr/bin/env bash
# D1 compact FA actual-enabled-shape correctness gate, physical card1.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/d1/fa-batch-gate-1"
mkdir -p "$OUT"
export B70_AGENT=D1 B70_GPU_LOCK_TIMEOUT=1800
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-before.txt" 2>&1 || exit 3
"$R/vllm/flashnext/harness/xpu-run.sh" --card 1 --memory 8g \
    --name vllm-fn-d1-fa-batch \
    --env "PYTHONPATH=$B/c7b-fa-overlay:$B/vllm" \
    --env PYTHONUNBUFFERED=1 --env SYCL_CACHE_PERSISTENT=0 \
    -- /opt/venv/bin/python "$R/vllm/flashnext/harness/d1_fa_batch_gate.py" \
    > "$OUT/gate.log" 2>&1
rc=$?
docker stop -t 10 vllm-fn-d1-fa-batch >/dev/null 2>&1 || true
"$R/bin/gpu-run" --card 1 "$R/bin/xpu-health" --card 1 > "$OUT/health-after.txt" 2>&1
health=$?
printf 'D1_BATCH_GATE_RC:%s HEALTH_RC:%s\n' "$rc" "$health" | tee "$OUT/status.txt"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$health"
