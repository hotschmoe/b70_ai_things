#!/usr/bin/env bash
# P0.4: Qwen3.8-27B W8A8-gptq + off-shelf DSpark on vLLM 0.26.0.
# Clone of serve_qwen38_radixark_dspark.sh; target is W8A8-gptq not NVFP4.
# Applies the SpecForge readout fix (vllm/dflash/patches/v0260) and remaps
# the drafter architecture DSparkDraftModel -> Qwen3DSparkModel so the
# v0.26 registry does not route it to DeepSeek-V4.
#
# method=dspark (not dflash -- PRE.3). THINK_BUDGET=0 (V2 rejects it).
# GRAPH=0 default: GRAPH=1 CGRECLAIM=0 died mid-HE+ (LOOP 3).
# P2PACCESS stays 0. Do not overwrite models/files/qwen3.8-27b/w8a8-gptq.
#
#   GRAPH=0 SPECTOK=7 MAXLEN=131072 PORT=18080 NAME=qwen38_w8a8_dspark \
#     ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
#   bash vllm/dflash/serve_qwen38_w8a8_dspark.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_w8a8_dspark}"
PATCH="$REPO/vllm/dflash/patches/v0260"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

[ -f "$PATCH/dflash.py" ] || { echo "missing $PATCH/dflash.py"; exit 1; }
[ -f "$PATCH/drafter_config.json" ] || { echo "missing $PATCH/drafter_config.json"; exit 1; }
[ -f "$PATCH/compile_key_spectok_so.py" ] || { echo "missing $PATCH/compile_key_spectok_so.py"; exit 1; }
[ -f "$PATCH/compile_key_sitecustomize.py" ] || { echo "missing $PATCH/compile_key_sitecustomize.py"; exit 1; }

SPECTOK="${SPECTOK:-7}"
DRAFTER_REL="${DRAFTER_REL:-qwen3.8-27b/dflash-drafter-fp8-b70}"
HOST_DRAFT="$REPO/models/files/${DRAFTER_REL}"
[ -d "$HOST_DRAFT" ] || { echo "MISSING $HOST_DRAFT"; exit 1; }

export CKPT="${CKPT:-/models/qwen3.8-27b/w8a8-gptq}"
export SERVED="${SERVED:-qwen3.8-27b-W8A8-gptq-dspark${SPECTOK}}"
export NAME PORT="${PORT:-18080}"
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export TP="${TP:-2}" GRAPH="${GRAPH:-0}" MAXLEN="${MAXLEN:-131072}"
export UTIL="${UTIL:-0.90}" MAXSEQS="${MAXSEQS:-2}"
# B1: KV_FP8=1 -> fp8_e5m2 storage KV via KVDTYPE (LOOP 6 was a no-op).
# Default 0 = bf16. Uncalibrated fp8 can tank plus; this fire G1-gates.
export KV_FP8="${KV_FP8:-0}"
if [ "$KV_FP8" = 1 ]; then
  export KVDTYPE="${KVDTYPE:-fp8_e5m2}"
  echo "=== KV_FP8=1 -> --kv-cache-dtype $KVDTYPE ===" >&2
else
  export KVDTYPE="${KVDTYPE:-}"
fi
# Do NOT set B70_NOMTP=1 -- that clears SPEC.
export B70_NOMTP=0
# V2 runner (forced by method=dspark) rejects thinking_token_budget.
export THINK_BUDGET="${THINK_BUDGET:-0}"
# MTPTOK may be reset to 3 by serve.sh ${MTPTOK:-3}; SPEC wins in lib.sh.
export SPEC="${SPEC:-{\"method\":\"dspark\",\"model\":\"/models/${DRAFTER_REL}\",\"num_speculative_tokens\":${SPECTOK}}}"
export P2PACCESS="${P2PACCESS:-0}"
# Compile-key SPECTOK + mounted _xpu_C SO (LOOP 26 / D2 D3). First
# PYTHONPATH sitecustomize; chains push-AR -> mtp_shim. Last -e wins.
export B70_EXTRA_MOUNTS="${B70_EXTRA_MOUNTS:+$B70_EXTRA_MOUNTS }${PATCH}/dflash.py:/workspace/vllm/vllm/v1/spec_decode/dflash.py:ro ${PATCH}/utils.py:/workspace/vllm/vllm/v1/spec_decode/utils.py:ro ${PATCH}/drafter_config.json:/models/${DRAFTER_REL}/config.json:ro ${PATCH}/compile_key_sitecustomize.py:/opt/compile_key_shim/sitecustomize.py:ro ${PATCH}/compile_key_spectok_so.py:/opt/compile_key_shim/compile_key_spectok_so.py:ro"
export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }PYTHONPATH=/opt/compile_key_shim:/opt/push_ar:/opt/mtp_shim"

echo "=== P0.4 W8A8-gptq + DSpark k=$SPECTOK  GRAPH=$GRAPH  maxlen=$MAXLEN  name=$NAME ==="
echo "=== method=dspark THINK_BUDGET=$THINK_BUDGET SERVED=$SERVED P2PACCESS=$P2PACCESS ==="
exec bash "$REPO/rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh" start
