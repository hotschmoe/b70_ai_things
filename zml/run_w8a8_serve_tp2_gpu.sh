#!/usr/bin/env bash
# zml/run_w8a8_serve_tp2_gpu.sh -- M5: end-to-end W8A8 qwen3.6-27b across BOTH B70s (TP=2). The
# W8A8 text-model weights are 32.6 GiB (17.5 i8 + 15.1 bf16) > one card's 30.3 GiB usable, so the
# 27B does NOT fit one card (single-card OOMs) -- TP=2 halves per-card weights to ~16 GiB.
#
# ATTENDED ONLY -- the TP=2 BCS/oneCCL wedge is reboot-only (see CLAUDE.md GPU Discipline,
# w8a8-mtp-enforce-eager-and-tp2-wedge, zml/ZML_TP_ALLREDUCE.md). GuC firmware pinned 70.54.0.
# Run under the gpu-run lease (BOTH cards):
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash zml/run_w8a8_serve_tp2_gpu.sh
#
# TP=2 is automatic: both cards visible (level_zero:gpu) -> Shardings.init makes a 2-way mesh and the
# model's .model partition tags shard q/k/v (dout), o (d), gate/up (dout), down (dout=intermediate,
# row-parallel). The W8A8 down_proj act-quant reduce-max runs over the SHARDED intermediate axis ->
# per-shard quant + dequant-before-allreduce (verify coherence; see ZML_TP_ALLREDUCE.md).
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
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"   # CRITICAL: avoid the P2P-in-serve wedge + override oneapi.zig:33 garbage
echo "=== zml W8A8 27B TP=2 serve  model=$MODEL seqlen=$SEQLEN topk=$TOPK  $(date) ==="
set +e   # ensure bazelisk shutdown + post-health run even on crash (don't leak the gpu-run flock)
"$BAZELISK" run //examples/llm --config=release \
  --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
  -- --model="$MODEL" --prompt="$PROMPT" --seqlen="$SEQLEN" --topk="$TOPK"
rc=$?
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== llm exit rc=$rc ; POST xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -3 || echo "[!] box may be WEDGED -- bin/xe-reset (reboot-only on this box)"
exit $rc
