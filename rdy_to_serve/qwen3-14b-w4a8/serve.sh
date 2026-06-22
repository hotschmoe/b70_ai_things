#!/usr/bin/env bash
# Qwen3-14B W4A8 (int4 weights / int8 activations, GPTQ, offline-prepacked) -- int8-activation 14B.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (PIECEWISE capture), wait healthy, gen-probe
#   bash serve.sh stop                            # stop + release the GPU
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + concurrency sweep + stop, one lease
#
# [!] IMAGE: vllm-xpu-env:int8g (our INT8 W8A8 kernel + register_fake; build images/int8g/). The int4
#     weights run through the W4A8 scheme; int8 activations light the systolic path.
# PREPACK: the weights are int4-packed on disk -> mount the patched loader (mixed_precision/xpu.py) + the
# W4A8 scheme (compressed_tensors_w4a8_int.py) and set VLLM_W4A8_PREPACKED=1, so vLLM loads the small packed
# weights directly (no large unpacked-int8 GPU transient). Dense 14B (Qwen3) -> no GDN, no vision tower.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:int8g}"
export CKPT="${CKPT:-/models/Qwen3-14B-W4A8-gptq-prepacked}"
export SERVED="${SERVED:-qwen3-14b-w4a8-gptq}"
export DTYPE="${DTYPE:-float16}"
export GRAPH="${GRAPH:-1}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-32}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32}"

# PREPACK: patched loader + W4A8 scheme (mounted over vLLM) + the env flag.
KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
MOUNTS=( -v "$SCRIPT_DIR/patches/xpu.py:$KP:ro" -v "$SCRIPT_DIR/patches/compressed_tensors_w4a8_int.py:$SP:ro" )
DOCKER_ENV=( -e VLLM_W4A8_PREPACKED=1 )

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
