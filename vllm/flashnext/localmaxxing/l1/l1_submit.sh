#!/usr/bin/env bash
# L1: submit already dry-run-validated payloads, one per 65 s (site limit 1/min).
# usage: l1_submit.sh <dir-with-payload_*.json> [suite...]   (auth: lmx saved config)
set -euo pipefail
D=$1; shift
SUITES=("${@:-reasoning-v1 code-v1}")
LMX=${LMX_BIN:-$HOME/.local/bin/lmx}
first=1
for s in ${SUITES[@]}; do
  P=$D/payload_$s.json
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['valid'] is True and d['verification']['verified'] is True, d" "$D/dry-run_$s.json"
  test -e "$D/submit_$s.json" && { echo "already submitted $s"; continue; }
  [ $first = 1 ] || sleep 65
  first=0
  date -u +%FT%TZ > "$D/submit_$s.time"
  "$LMX" speed-test submit "$P" --json > "$D/submit_$s.json" 2>&1 || { echo "submit $s failed"; cat "$D/submit_$s.json"; exit 1; }
  echo "$s submitted"; head -c 600 "$D/submit_$s.json"; echo
done
