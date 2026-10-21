#!/bin/bash
# Separate Strata reference process. Does not start or alter hotschmoe-dd.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
S=/mnt/vm_8tb/b70/build/strata-ab
OUT=$B/x6
NAME=vllm-fn-x6-strata-reference
export B70_AGENT=X6 B70_GPU_LOCK_TIMEOUT=21600
python3 "$R/vllm/flashnext/harness/x6_priority_wait.py"
"$R/bin/gpu-run" "$R/bin/xpu-health" > "$OUT/strata-health-before.txt" 2>&1 || exit 3
cleanup() { docker stop -t 15 "$NAME" >/dev/null 2>&1 || true; kill "${WD:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
( while true; do
    avail=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
    if [ "$avail" -lt 8388608 ]; then echo ABORT_LOW_RAM; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 1; fi
    sleep 2
  done ) &
WD=$!
"$R/bin/gpu-run" docker run --rm --init --name "$NAME" --network none \
  --memory 90g --device /dev/dri --ulimit memlock=-1 --cap-add SYS_PTRACE \
  -v "$R:$R:ro" -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 -v /mnt/cache/b70:/mnt/cache/b70:ro \
  -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e ZES_ENABLE_SYSMAN=1 -e STRATA_VERIFY_DEVICE_PLAN=1 -e STRATA_VERIFY_NO_HOST=1 \
  -e STRATA_MIRROR_MIB=0 -e STRATA_STAGE_TRIM=1 -e STRATA_KV_WIDE_HOST=0 -e STRATA_STAGER_THREADS=12 \
  --entrypoint bash sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2 -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec timeout --signal=TERM --kill-after=20 8m /mnt/vm_8tb/b70/build/strata-ab/venv/bin/python /mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/x6_strata_reference.py --config /mnt/vm_8tb/b70/build/vllm-flashnext/x6/strata-reference-config.json --prompts /mnt/vm_8tb/b70/build/vllm-flashnext/x6/reference-prompts.json --out /mnt/vm_8tb/b70/build/vllm-flashnext/x6/strata-references.json' > "$OUT/strata-reference.log" 2>&1
rc=$?
cleanup
"$R/bin/gpu-run" "$R/bin/xpu-health" > "$OUT/strata-health-after.txt" 2>&1
health=$?
printf 'STRATA_RC:%s HEALTH_RC:%s\n' "$rc" "$health"
exit "$rc"
