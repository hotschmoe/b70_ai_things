#!/usr/bin/env bash
# Parent schedules this after D1 component gate PASS and source freeze.
# Both cards; one control and one native FA-plan16 arm; no submission.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H="$R/vllm/flashnext/harness"
OUT="$B/d1/matched-r1"
mkdir -p "$OUT"
[ ! -e "$OUT/started" ] || { echo D1_MATCHED_ALREADY_STARTED; exit 2; }
touch "$OUT/started"
B70_AGENT=D1 "$R/bin/cpu-run" python3 "$H/d1_source_identity.py" snapshot \
  "$OUT/identity.json" > "$OUT/identity-before.log" 2>&1 || exit 3
export MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export F1_LENGTHS=8192 F1_COHERENCE_LEN=8192 F1_REPEAT=2 F1_DECODE=0
export RUN_PROMO_GATE=1 SKIP_HEALTH_BEFORE=1
export F1_API_WRAPPER="$H/d1_api.py"
export F1_POST_PROBE_SCRIPT="$H/d1_combined_callback.sh"
for arm in control plan16; do
  export TAG="d1-$arm-r1"
  if [ "$arm" = control ]; then
    export F1_PYPATH_PRE=""
    export O4_SERVE_EXTRA='B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=0'
  else
    export F1_PYPATH_PRE="$B/d1/fa-overlay"
    export O4_SERVE_EXTRA='B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=16'
  fi
  bash "$H/f1_merged_serve.sh" > "$OUT/$arm-launch.log" 2>&1
  rc=$?
  printf 'D1_ARM:%s RC:%s\n' "$arm" "$rc" | tee "$OUT/$arm-status.txt"
  B70_AGENT=D1 "$R/bin/cpu-run" python3 "$H/d1_source_identity.py" verify \
    "$OUT/identity.json" > "$OUT/$arm-identity-after.log" 2>&1 || exit 4
  AOUT="$B/f1/merged-$TAG"
  grep -q 'xpu-health: HEALTHY' "$AOUT/health-after.txt" || exit 5
  grep -q 'SERVE_RC:0' "$AOUT/window.log" || exit 6
  python3 - "$AOUT/greedy.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
assert rows[0]['first_ok'] and all(row['match'] for row in rows[1:])
PY
  [ "$?" = 0 ] || exit 7
  # Strict repeat/gate failures remain recorded, but do not discard the
  # second diagnostic arm when coherence/teardown/health/source are sound.
done
echo D1_MATCHED_WINDOWS_DONE
