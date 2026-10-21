#!/usr/bin/env bash
# HTTP clients only; invoked inside the owner's merged-serve window.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ARGS=()
[ -n "${N1_REFERENCE:-}" ] && ARGS+=(--reference "$N1_REFERENCE")
B70_AGENT=N1 "$REPO/bin/cpu-run" --quiet python3 \
  "$REPO/vllm/flashnext/harness/n1_repeat_http.py" \
  --base-url "${F1_BASE:?}" --expect-alias "${F1_ALIAS:?}" \
  --out "${F1_OUT_DIR:?}/n1-repeat.json" "${ARGS[@]}" > "$F1_OUT_DIR/n1-repeat.log" 2>&1
RC=$?
tail -n 1 "$F1_OUT_DIR/n1-repeat.log"
sha256sum -c "$F1_OUT_DIR/n1-before.sha256" > "$F1_OUT_DIR/n1-after-check.txt" 2>&1
SRC=$?
echo "N1_REPEAT_RC:$RC N1_SOURCE_RC:$SRC"
[ "$RC" -eq 0 ] && [ "$SRC" -eq 0 ]
