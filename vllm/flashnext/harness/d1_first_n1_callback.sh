#!/bin/bash
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
printf 'detail\n' > "${F1_OUT_DIR:?}/d1-liveness/arm"
B70_AGENT=D1 "$REPO/bin/cpu-run" --quiet python3 "$D/d1_first_n1_callback.py" \
  --base-url "${F1_BASE:?}" --expect-alias "${F1_ALIAS:?}" \
  --out "$F1_OUT_DIR/d1-first-n1-callback.json" \
  > "$F1_OUT_DIR/d1-first-n1-callback.log" 2>&1
