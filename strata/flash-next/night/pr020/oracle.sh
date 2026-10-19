#!/bin/bash
# #2526 slot parking (no MTP; layer split turns batch MTP off) and #2511 draft accounting, each vs main020.
NL=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/night/nl.py
C=/mnt/vm_8tb/b70/build/strata-ab/pr020-check
R=/mnt/vm_8tb/b70/results/pr-020/runs
while read name cfg mode; do
  echo "== $(date +%T) $name $mode"
  python3 $NL start $name $C/configs/$cfg.json || { echo "START FAILED $name"; python3 $NL stop $name; continue; }
  log=$(python3 -c "import json;print(json.load(open('$C/configs/$cfg.json'))['log'])")
  python3 $C/slot_oracle.py $mode --log $log --out $R/oracle-$name 2>&1 | tail -25
  python3 $NL stop $name 2>&1 | tail -3
done <<'L'
main020-park main020-c2-nomtp park
pr2526-park pr2526-020-c2-nomtp park
main020-drafts main020-c2 drafts
pr2511-drafts pr2511-020-c2 drafts
L
echo ORACLE_DONE
