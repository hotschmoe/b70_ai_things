#!/usr/bin/env bash
# ROOT ONLY after CPU reference, source/model/build audit and announced card1 window.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REFS=$(realpath "${1:?completed CPU reference directory}") || exit 2
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$HERE/run-clean"
mkdir "$OUT" || { echo 'FAIL preserve prior attempt; no implicit retry';exit 2; }
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=S80-grouped-prefill \
 "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 1 \
 env ZE_AFFINITY_MASK=1 ONEAPI_DEVICE_SELECTOR='level_zero:*' \
 bash "$HERE/leased_run.sh" "$OUT" "$REFS" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[ "${rc[1]}" = 0 ] || exit "${rc[1]}"
exit "${rc[0]}"
