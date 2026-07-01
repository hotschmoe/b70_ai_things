#!/usr/bin/env bash
# zml/run_w8a8_mtp_drive_tp2_gpu.sh -- Step 3-4: NEXTN/MTP DRIVE (real speculative decode) on the W8A8
# qwen3.6-27b across BOTH B70s (TP=2). ZML_MTP_MEASURE=1 makes the session run the MTP head as an
# OBSERVER alongside the normal greedy decode: each step it drafts the next token from (committed
# token, its post-final-norm hidden, position p) and scores the previous draft against the token the
# main model actually produces. The main model still drives (output unchanged); this only measures
# how often the draft matches. Target accept ~0.84 (validates the head math end-to-end on GPU).
#
# ATTENDED ONLY -- TP=2 BCS/oneCCL wedge is reboot-only (CLAUDE.md GPU Discipline). GuC 70.54.0.
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash zml/run_w8a8_mtp_drive_tp2_gpu.sh
set -uo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"
MODEL="${MODEL:-$REPO/models/files/qwen3.6-27b/w8a8-sqgptq}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one short sentence.}"
SEQLEN="${SEQLEN:-1024}"
TOPK="${TOPK:-1}"

echo "=== PRE xpu-health (both cards) ===" && "$REPO/bin/xpu-health" 2>&1 | tail -2
cd "$ZML"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"   # BOTH cards -> TP=2
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"   # avoid the P2P-in-serve wedge + oneapi.zig:33 garbage
export ZML_MTP=1                                                 # <-- MTP DRIVE mode (spec-decode loop)
echo "=== zml W8A8 27B TP=2 MTP-DRIVE  model=$MODEL seqlen=$SEQLEN topk=$TOPK  $(date) ==="
set +e
"$BAZELISK" run //examples/llm --config=release \
  --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
  -- --model="$MODEL" --prompt="$PROMPT" --seqlen="$SEQLEN" --topk="$TOPK"
rc=$?
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== llm exit rc=$rc ; POST xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -3 || echo "[!] box may be WEDGED -- bin/xe-reset (reboot-only on this box)"
exit $rc
