#!/usr/bin/env bash
# ROOT ONLY: reviewed source, root-attested S139 same-boot/no-intervening-GPU, announced card0 window.
# Export B70_DISCOVERY_REUSE_BOOT_ID=<known matching S139 boot> and B70_DISCOVERY_NO_INTERVENING_GPU=1.
set -uo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT="$P/run"
mkdir "$OUT" || { echo 'STOP preserve existing attempt; no retry'; exit 2; }
B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=UR-default-adapter-discovery \
  "$REPO/bin/cpu-run" --quiet "$REPO/bin/gpu-run" --card 0 \
  env ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' bash "$P/leased-run.sh" "$OUT" 2>&1 | tee "$OUT/driver.log"
rc=("${PIPESTATUS[@]}")
printf '%s\n' "${rc[0]}" > "$OUT/driver.exit"
printf '%s\n' "${rc[1]}" > "$OUT/tee.exit"
[[ ${rc[1]} == 0 ]] || exit "${rc[1]}"
exit "${rc[0]}"
