#!/usr/bin/env bash
# Campaign M1: RadixArk 3.8 NVFP4 + DSpark on vLLM 0.26.0 @ native 262k.
# Applies the SpecForge readout fix (vllm/dflash/patches/v0260) and remaps
# the drafter architecture DSparkDraftModel -> Qwen3DSparkModel so the
# v0.26 registry does not route it to DeepSeek-V4.
#
#   ./bin/gpu-run bash vllm/dflash/serve_qwen38_radixark_dspark.sh start
#   bash vllm/dflash/serve_qwen38_radixark_dspark.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_radixark_dspark}"
PATCH="$REPO/vllm/dflash/patches/v0260"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$PATCH/dflash.py" ] || { echo "missing $PATCH/dflash.py"; exit 1; }
[ -f "$PATCH/drafter_config.json" ] || { echo "missing $PATCH/drafter_config.json"; exit 1; }

SPECTOK="${SPECTOK:-7}"
DRAFTER_REL="${DRAFTER_REL:-qwen3.8-27b/dflash-drafter-fp8-b70}"
export NAME PORT="${PORT:-8078}"
export TP="${TP:-2}" GRAPH="${GRAPH:-1}" MAXLEN="${MAXLEN:-262144}"
# DSpark draft KV is extra vs MTP: first 262k try at UTIL=0.85 needed 13.25
# GiB and only had 9.81 (est max 190528). Raise util and drop seqs.
export UTIL="${UTIL:-0.90}" MAXSEQS="${MAXSEQS:-2}"
# bf16 KV + DSpark draft state needs 13.25 GiB for 262k; 0.92 util
# only freed 12.29 (est max 242112). fp8 KV is the 262k lever.
export KV_FP8="${KV_FP8:-1}"
# V2 runner (forced by method=dspark) rejects thinking_token_budget.
export THINK_BUDGET="${THINK_BUDGET:-0}"
export MTPTOK=   # disable NEXTN; DSpark is SPEC
# method=dspark (not dflash): v0.26 forces the V2 runner and uses the
# sample-from-anchor speculator. method=dflash wraps the arch as
# DFlashQwen3DSparkModel which is not registered.
export SPEC="${SPEC:-{\"method\":\"dspark\",\"model\":\"/models/${DRAFTER_REL}\",\"num_speculative_tokens\":${SPECTOK}}}"
export SERVED="${SERVED:-qwen3.8-27b-NVFP4-radixark-dspark${SPECTOK}}"
export CAPSIZES="${CAPSIZES:-1,2,4,8}"
export B70_EXTRA_MOUNTS="${B70_EXTRA_MOUNTS:+$B70_EXTRA_MOUNTS }${PATCH}/dflash.py:/workspace/vllm/vllm/v1/spec_decode/dflash.py:ro ${PATCH}/utils.py:/workspace/vllm/vllm/v1/spec_decode/utils.py:ro ${PATCH}/drafter_config.json:/models/${DRAFTER_REL}/config.json:ro"

echo "=== M1 RadixArk + DSpark k=$SPECTOK  GRAPH=$GRAPH  maxlen=$MAXLEN  name=$NAME ==="
exec bash "$REPO/rdy_to_serve/vllm/qwen38-27b-nvfp4/serve.sh" start
