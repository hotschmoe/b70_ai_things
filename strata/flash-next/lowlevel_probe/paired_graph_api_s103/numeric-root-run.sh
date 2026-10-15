#!/usr/bin/env bash
# Untimed numerical/API count probe: deliberately no quiet CPU lease.
set -uo pipefail
P=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/paired-graph-s103-prep-v2
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$P/run"
mkdir "$OUT" || { echo 'STOP preserve existing attempt; no retry'; exit 2; }
B70_AGENT=S103-graph-API-numeric "$REPO/bin/gpu-run" --card 0 env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' bash "$P/leased-run.sh" "$OUT" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[[ ${rc[1]} == 0 ]] || exit "${rc[1]}"
exit "${rc[0]}"
