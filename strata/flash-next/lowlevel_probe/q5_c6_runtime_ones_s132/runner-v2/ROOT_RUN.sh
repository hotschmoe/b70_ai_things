#!/usr/bin/env bash
# ROOT ONLY after full source/build + explicit root RUN_GO + announced card0 ~60s window.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=${1:?new disk output directory required};shift
mkdir "$OUT" || { echo 'FAIL preserve prior attempt; no automatic retry';exit 2; }
OUT=$(realpath "$OUT")
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=Q5-HEAD-C6-RUNTIME-ONES \
 "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 \
 env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' \
 python3 "$HERE/leased_run.py" --out "$OUT" "$@" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[ "${rc[1]}" = 0 ] || exit "${rc[1]}"
exit "${rc[0]}"
