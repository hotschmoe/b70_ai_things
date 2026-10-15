#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
UNITRACE=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build/unitrace
[[ -x "$UNITRACE" && -x /probe/paired_graph_s103 && -e /dev/dri ]]
cd /probe
sha256sum -c /probe/runtime.sha256
for mode in 0 1; do
  mkdir "/out/mode$mode"
  set +e
  "$UNITRACE" -h -d -v -o "/out/mode$mode/unitrace.txt" /probe/paired_graph_s103 "$mode" "/out/mode$mode/leaf" > "/out/mode$mode/leaf.log" 2> "/out/mode$mode/leaf.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "/out/mode$mode/leaf.exit"
  [[ $rc == 0 ]] || exit "$rc"
  python3 /src/analyze.py /out "$mode" > "/out/mode$mode/analysis.stdout" 2> "/out/mode$mode/analysis.stderr"
done
python3 /src/analyze.py /out > /out/analysis.stdout 2> /out/analysis.stderr
