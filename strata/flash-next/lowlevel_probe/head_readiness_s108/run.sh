#!/usr/bin/env bash
set -uo pipefail
prep=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/head-readiness-s108
test ! -e "$prep/start.txt" || exit 64
date -u +%FT%TZ > "$prep/start.txt"
/usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S maxrss_kib=%M exit=%x' -o "$prep/time.txt" timeout --signal=TERM 180 python3 "$prep/extract.py" > "$prep/stdout.log" 2> "$prep/stderr.log"
code=$?
printf '%s\n' "$code" > "$prep/exit.txt"
date -u +%FT%TZ > "$prep/end.txt"
cat "$prep/time.txt"
cat "$prep/stdout.log"
exit "$code"
