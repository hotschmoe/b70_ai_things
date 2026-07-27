#!/usr/bin/env bash
# Carry the torch-2.12 custom INT8 registry and oneCCL fix onto vLLM v0.26.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_SUFFIX=v0260
export VLLM_LABEL=0.26.0
export BASE="${BASE:-vllm-xpu-env:v0260}"
exec bash "$SCRIPT_DIR/bake_v0251.sh"
