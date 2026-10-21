#!/usr/bin/env bash
# Prepared only: schedule both-card fullmodel window after primitive PASS.
# N1_FLAG=0 gives a matched control; =1 enables only GDN b|a determinism.
# Root can prepend D1 overlay via F1_PYPATH_PRE and add O4_SERVE_EXTRA.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="$B/f1/merged-${TAG:?TAG}"
mkdir -p "$OUT"
export F1_PYPATH_PRE="${F1_PYPATH_PRE:+$F1_PYPATH_PRE:}$REPO/vllm/flashnext/plugins/b70_n1_xpu"
export O4_SERVE_EXTRA="${O4_SERVE_EXTRA:-} B70_GDN_BA_DETERMINISTIC=${N1_FLAG:-1} B70_N1_SO=$B/n1/_C.so"
export F1_POST_PROBE_SCRIPT="$REPO/vllm/flashnext/harness/n1_repeat_callback.sh"
export MAX_LEN="${MAX_LEN:-131072}" RUN_PROMO_GATE="${RUN_PROMO_GATE:-1}"
sha256sum "$B/n1/_C.so" "$B/vllm/vllm/models/qwen4_exp/xpu/model.py" \
  "$REPO/vllm/flashnext/plugins/b70_n1_xpu/b70_n1_xpu/__init__.py" > "$OUT/n1-before.sha256"
exec bash "$REPO/vllm/flashnext/harness/f1_merged_serve.sh"
