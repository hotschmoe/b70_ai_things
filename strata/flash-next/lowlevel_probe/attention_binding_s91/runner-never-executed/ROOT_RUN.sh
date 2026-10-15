#!/usr/bin/env bash
# ROOT ONLY after resolved source reviews + actual build audit + announced card0 lease.
set -uo pipefail
[ "${B70_S91_REVALIDATED_S90_PREHEALTH:-}" = 1 ] || { echo "Root must revalidate recent S90 prehealth before launch";exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$HERE/run-clean"
mkdir "$OUT" || { echo 'FAIL preserve prior attempt; no implicit retry';exit 2; }
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=S91-attn-score \
  "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 \
  env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' \
  bash "$HERE/leased_run.sh" "$OUT" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[ "${rc[1]}" = 0 ] || exit "${rc[1]}"
exit "${rc[0]}"
