#!/usr/bin/env bash
# zml/run_w8a8_serve_gpu.sh -- end-to-end W8A8 qwen3.6-27b text generation on ONE B70 (ZML_W8A8.md
# M3/M4 follow-up: the full-model serve). SINGLE CARD (level_zero:0); the W8A8 weights are ~27 GB so
# they fit one 32 GB B70 (bf16 27B = 54 GB would NOT -- that needs TP=2/M5). No collectives -> does
# not touch the TP=2 wedge path. Daily driver must be DOWN; run under the gpu-run lease:
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run --card 0 bash zml/run_w8a8_serve_gpu.sh
#
# QuantizedLinear auto-selects int8 (the checkpoint carries weight_scale); GDN/vision/MTP/lm_head
# stay bf16 (ignore list). vision+MTP are not ported in the zml qwen3_5 model -> text-only serve.
set -uo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"
MODEL="${MODEL:-$REPO/models/files/qwen3.6-27b/w8a8-sqgptq}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one sentence.}"
SEQLEN="${SEQLEN:-2048}"
TOPK="${TOPK:-1}"          # greedy for a deterministic coherence check
DUMP="${DUMP:-}"          # set DUMP=1 to also dump StableHLO (act-quant fusion / down_proj reduce check)

echo "=== pre-flight xpu-health ===" && "$REPO/bin/xpu-health" 2>&1 | tail -2
cd "$ZML"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"
if [[ -n "$DUMP" ]]; then
  export XLA_FLAGS="--xla_dump_to=/mnt/vm_8tb/b70/w8a8_serve_hlo --xla_dump_hlo_as_text"
  export ONEDNN_VERBOSE="${ONEDNN_VERBOSE:-dispatch}"
fi
echo "=== zml W8A8 27B serve  model=$MODEL  seqlen=$SEQLEN topk=$TOPK  $(date) ==="
set +e
"$BAZELISK" run //examples/llm --config=release \
  --@zml//platforms:cpu=false --@zml//platforms:oneapi=true \
  -- --model="$MODEL" --prompt="$PROMPT" --seqlen="$SEQLEN" --topk="$TOPK"
rc=$?
"$BAZELISK" shutdown >/dev/null 2>&1 || true
echo "=== llm exit rc=$rc ; post-run xpu-health ==="
"$REPO/bin/xpu-health" 2>&1 | tail -2 || echo "[!] box may be wedged -- bin/xe-reset"
exit $rc
