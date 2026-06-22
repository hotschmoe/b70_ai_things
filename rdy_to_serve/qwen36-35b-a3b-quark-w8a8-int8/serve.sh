#!/usr/bin/env bash
# Qwen3.6-35B-A3B Quark W8A8 INT8 (TRUE int8 MoE) -- 2x B70, TP=2. See ./README.md for the full recipe.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (TP=2, eager), wait healthy, gen-probe, stay up
#   bash serve.sh stop                            # stop + release the GPU
#   bash serve.sh bench                           # concurrency sweep vs the running server
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + bench + stop, PIECEWISE graph capture
#
# IMAGE: vllm-xpu-env:v0230 (vLLM 0.23.0). NEVER llm-scaler 0.14.x (no _moe_C -> int8 MoE hard-fails).
# THE ONE PATCH: patches/quark.py reroutes the int8 LINEAR layers (linear_attn.*, mlp.shared_expert.*)
# to a weight-only int8->bf16 dequant GEMM (XPU has no int8 scaled-mm kernel). The 256 routed experts
# stay TRUE int8 via the Triton fused_moe_kernel. Pure-Python patch -> bind-mounted per-container
# (mount-not-bake: it cannot affect any other model's container). See ORGANIZATION.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0230}"
export CKPT="${CKPT:-/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"
export SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
export QUANT="${QUANT:-quark}"
export TP="${TP:-2}"                        # int8 weights ~35 GB -> 17.5 GiB/card; does NOT fit one card
export GRAPH="${GRAPH:-0}"                  # eager works at all conc; GRAPH=1 = PIECEWISE capture lever
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.92}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-8}"
export NOMM="${NOMM:-1}"                    # text-only VLM serve (skip vision encoder)

# Bind-mount the one Python patch over BOTH possible vLLM locations in the image (workspace + venv).
PATCH="$SCRIPT_DIR/patches/quark.py"
[ -f "$PATCH" ] || { echo "[!] missing patch: $PATCH"; exit 2; }
Q1=/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark.py
Q2=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py
# NOTE: arrays cannot be `export`ed in bash; serve.sh sources lib.sh (same shell) so a plain array is visible.
MOUNTS=( -v "$PATCH:$Q1:ro" -v "$PATCH:$Q2:ro" )

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
