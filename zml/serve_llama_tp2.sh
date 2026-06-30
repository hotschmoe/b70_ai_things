#!/usr/bin/env bash
# zml/serve_llama_tp2.sh -- first dense-LLM milestone on ZML/oneAPI: a small Llama TP=2 across both B70s.
# Run ONLY after test_sharding.sh is green. This is the realistic ZML LLM target today -- NOT qwen3.6
# (which needs a multi-week Zig port: model_type detection + vision tower + MTP head; see REVIEW sec 4).
#
# ZML runs bf16/f16 via XLA. There is no quantized path -> this is the "bf16 dense, sharded TP=2" mapping
# of the request, a compiler/TP architecture experiment, not a daily-driver replacement for sglang W8A8.
#
# MUST run under the GPU lease (both cards):
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash zml/serve_llama_tp2.sh
#
# Knobs: MODEL (hf:// id), PROMPT, TOPK. Default Llama-3.2-1B-Instruct (README's own example; smallest).
set -euo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"
MODEL="${MODEL:-hf://meta-llama/Llama-3.2-1B-Instruct}"
PROMPT="${PROMPT:-Say hello in one sentence.}"
TOPK="${TOPK:-1}"

echo "=== pre-flight xpu-health ===" && "$REPO/bin/xpu-health" 2>&1 | tail -2
cd "$ZML"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"   # override oneapi.zig:33 garbage default (see test_sharding.sh)
echo "=== zml oneAPI LLM TP=2  model=$MODEL  $(date) ==="
"$BAZELISK" run //examples/llm \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true \
  -- \
  --model="$MODEL" \
  --topk="$TOPK" \
  --prompt="$PROMPT"
rc=$?
# Shut down the bazel DAEMON before returning. CRITICAL: when bazelisk runs under the gpu-run flock,
# the persistent bazel server inherits the lock fds and keeps the GPU lease HELD for ~3h after this
# script exits -- blocking every later gpu-run (incl. the daily-driver restore). Shutting it down here
# releases the inherited fds while we still hold the lease.
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== llm exit rc=$rc ; post-run xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -2 || echo "[!] box may be wedged -- bin/xe-reset"
exit $rc
