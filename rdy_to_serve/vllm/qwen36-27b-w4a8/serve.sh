#!/usr/bin/env bash
# Qwen3.6-27B W4A8 (int4 weights / int8 activations, SmoothQuant+GPTQ, prepacked, GDN+lm_head bf16).
# Self-contained recipe; shared plumbing in ../_common/lib.sh.  [SECONDARY: w4a16 is faster -- see
# ../qwen36-27b-int4/. Use THIS only for the int8-activation / int8-XMX path on the 27B.]
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (PIECEWISE capture), wait healthy, gen-probe
#   bash serve.sh stop                            # stop + release the GPU
#
# [!] IMAGE: vllm-xpu-env:int8g (build images/int8g/). Two extra needs beyond the 14B W4A8:
#  (1) PREPACK loader + W4A8 scheme (int4-packed weights) + VLLM_W4A8_PREPACKED=1.
#  (2) GDN: Qwen3.6-27B uses gated-delta-net; the :int8g baked kernel ships GDN_KERNELS_ENABLED=OFF
#      (-> "_xpu_C has no attribute gdn_attention" at decode). Mount the GDN-enabled _xpu_C.abi3.so (+
#      sibling libgdn_attn_kernels_xe_2.so) from the host kernel build over the baked one. This is a
#      large compiled binary -> referenced by host path (not copied into the dir), per ORGANIZATION.md.
# Served with fp8 KV (VRAM-tight, ~24 GiB). Decode ~20.9 t/s captured (< w4a16's ~30.8).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/mnt/vm_8tb/b70}"     # needed to reference the host GDN .so before sourcing lib.sh

export IMG="${IMG:-vllm-xpu-env:int8g}"
export CKPT="${CKPT:-/models/qwen3.6-27b/w4a8-sqgptq}"
export SERVED="${SERVED:-qwen36-27b-w4a8-sqgptq}"
export DTYPE="${DTYPE:-auto}"
export GRAPH="${GRAPH:-1}"
export NOMM="${NOMM:-1}"                    # 27B is a qwen3_5 VLM -> text-only
# fp16 KV: vLLM 0.23 rejects fp8 KV on this checkpoint ("fp8_e5m2 kv-cache is not supported with fp8
# checkpoints"). Override KVDTYPE=fp8_e5m2 only on an image/vLLM where it is accepted.
export KVDTYPE="${KVDTYPE:-}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-32}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32}"

KP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py
SP=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"        # GDN-enabled kernel
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
MOUNTS=( -v "$SCRIPT_DIR/patches/xpu.py:$KP:ro"
         -v "$SCRIPT_DIR/patches/compressed_tensors_w4a8_int.py:$SP:ro"
         -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro"
         -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
DOCKER_ENV=( -e VLLM_W4A8_PREPACKED=1 )

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
