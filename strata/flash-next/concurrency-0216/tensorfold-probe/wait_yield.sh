#!/usr/bin/env bash
# usage: wait_yield.sh SECONDS ; exits 0 when a GPU YIELDED line appears after the 05:10 YIELD REQUEST line
O=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/ORCHESTRATOR.md
end=$(( $(date +%s) + $1 ))
while [ $(date +%s) -lt $end ]; do
  req=$(grep -n 'GPU YIELD REQUEST' $O | tail -1 | cut -d: -f1)
  y=$(grep -nE '^- .*GPU YIELDED( at)? +[0-9]' $O | tail -1 | cut -d: -f1)
  if [ -n "$y" ] && [ -n "$req" ] && [ "$y" -gt "$req" ]; then echo "YIELDED: $(sed -n "$(echo $y)p" $O | cut -c1-200)"; exit 0; fi
  sleep 120
done
echo "not yet $(date -u +%T)"; exit 1
