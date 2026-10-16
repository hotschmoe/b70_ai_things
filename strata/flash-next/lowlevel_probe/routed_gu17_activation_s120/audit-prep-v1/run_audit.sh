#!/usr/bin/env bash
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/routed-gu17-existing-audit
[[ ! -e "$OUT" ]]
cd "$REPO"
exec env B70_AGENT=GU17-existing-selected-audit B70_CPU_LOCK_TIMEOUT=1 bin/cpu-run timeout 60 python3 "$P/audit.py" "$OUT"
