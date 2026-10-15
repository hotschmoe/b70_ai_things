#!/usr/bin/env bash
# ROOT ONLY after actual source/model review + build audit + announced card0 lease.
set -uo pipefail
HERE=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-dense-oracle-s65-prep
EXEC_DIR=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-dense-s65-execution
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$HERE/run-clean"
mkdir "$OUT" || { echo 'FAIL preserve prior attempt; no implicit retry';exit 2; }
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=S65-IQ3S-SWAR \
  "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 \
  env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' \
  bash "$EXEC_DIR/leased_run.sh" "$OUT" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[ "${rc[1]}" = 0 ] || exit "${rc[1]}"
exit "${rc[0]}"
