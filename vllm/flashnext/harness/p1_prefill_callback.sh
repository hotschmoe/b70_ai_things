#!/usr/bin/env bash
# Called inside f1_merged_serve.sh; this callback owns its quiet CPU window.
set -euo pipefail
repo=/mnt/vm_8tb/github/b70_ai_things
: "${F1_BASE:?}" "${F1_ALIAS:?}" "${F1_OUT_DIR:?}"
: "${B70_PREFILL_SELECTED_DQ:?}" "${P1_PREFILL_FIXTURES:?}"
B70_AGENT=P1 B70_CPU_LOCK_TIMEOUT=900 "$repo/bin/cpu-run" --quiet \
  python3 "$repo/vllm/flashnext/harness/p1_prefill_probe.py" \
  --base-url "$F1_BASE" --alias "$F1_ALIAS" \
  --fixtures "$P1_PREFILL_FIXTURES" --arm "$B70_PREFILL_SELECTED_DQ" \
  --out "$F1_OUT_DIR/p1-prefill.json" > "$F1_OUT_DIR/p1-prefill.log" 2>&1
