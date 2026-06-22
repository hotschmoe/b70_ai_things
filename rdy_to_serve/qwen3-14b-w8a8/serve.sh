#!/usr/bin/env bash
# Qwen3-14B W8A8 (true INT8, compressed-tensors / AutoRound) -- the int8-kernel baseline.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (PIECEWISE capture), wait healthy, gen-probe
#   bash serve.sh stop                            # stop + release the GPU
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + concurrency sweep + stop, one lease
#
# [!] IMAGE: vllm-xpu-env:int8g = :int8 (our oneDNN INT8 W8A8 GEMM, contrib/vllm_int8_xpu, registered as
#     XPUInt8ScaledMMLinearKernel) + register_fake so XPU graph capture can trace the custom int8 ops.
#     This is THE real low-precision compute path on the B70 (Xe2 has no native FP8). Build: images/int8g/.
# vLLM auto-detects the compressed-tensors W8A8 int8 scheme from config -> "Selected
# XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8" (no --quantization flag needed).
# Dense 14B (Qwen3, not Qwen3.6) -> no GDN, no prepack, no vision tower. ~16 GiB, fits ONE 32 GB B70.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:int8g}"
export CKPT="${CKPT:-/models/Qwen3-14B-W8A8-autoround}"
export SERVED="${SERVED:-qwen3-14b-w8a8-autoround}"
export DTYPE="${DTYPE:-float16}"           # 14B compute dtype; int8 W8A8 runs the linear layers true-int8
export GRAPH="${GRAPH:-1}"                  # PIECEWISE capture (int8g enables tracing the custom int8 ops)
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-32}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32}"
# tool-calling off by default (this is a quant baseline / eval target); enable with TOOLCALL=1 if serving agents.

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
