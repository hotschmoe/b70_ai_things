#!/usr/bin/env bash
# Always collect canonical diagnostics even when the preceding repeat fails.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H="$R/vllm/flashnext/harness"
OUT="${F1_OUT_DIR:?}"
sha256sum "$H/n1_repeat_http.py" "$H/promo_gate.py" \
  "$B/vllm/vllm/models/qwen4_exp/xpu/model.py" > "$OUT/d1-repeat-before.sha256"
ARGS=()
[ -n "${N1_REFERENCE:-}" ] && ARGS+=(--reference "$N1_REFERENCE")
B70_AGENT=D1 "$R/bin/cpu-run" --quiet python3 "$H/n1_repeat_http.py" \
  --base-url "${F1_BASE:?}" --expect-alias "${F1_ALIAS:?}" \
  --out "$OUT/n1-repeat.json" "${ARGS[@]}" > "$OUT/n1-repeat.log" 2>&1
repeat_rc=$?
sha256sum -c "$OUT/d1-repeat-before.sha256" > "$OUT/d1-repeat-after-check.txt" 2>&1
source_rc=$?
bash "$R/vllm/flashnext/harness/d1_lmx_callback.sh"
lmx_rc=$?
printf 'D1_REPEAT_RC:%s D1_SOURCE_RC:%s D1_LMX_RC:%s\n' "$repeat_rc" "$source_rc" "$lmx_rc" \
  | tee "${F1_OUT_DIR:?}/d1-callback-status.txt"
if [ "$repeat_rc" -ne 0 ]; then exit "$repeat_rc"; fi
if [ "$source_rc" -ne 0 ]; then exit "$source_rc"; fi
exit "$lmx_rc"
