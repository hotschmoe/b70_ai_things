#!/usr/bin/env bash
# ROOT ONLY after exact reviews, build audit and announced physicalcard1 window.
set -uo pipefail
P=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-more-groups-s73-prep
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$P/run-clean"
mkdir "$OUT" || exit 2
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=S73-MORE-ABBA "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 1 env ZE_AFFINITY_MASK=1 ONEAPI_DEVICE_SELECTOR='level_zero:*' bash "$P/leased_run.sh" "$OUT" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}");printf '%s\n' "${rc[0]}" > "$OUT/driver.exit";printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[ "${rc[1]}" = 0 ] || exit "${rc[1]}"
exit "${rc[0]}"
