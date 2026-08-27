#!/usr/bin/env bash
# Build the ABI-specific XPU operator extension and matching GDN sidecar needed
# by the v0.28 NVFP4 lane. The release image provides its other stock kernels.
set -euo pipefail

cd /build/vllm-xpu-kernels

export VLLM_VERSION_OVERRIDE=0.1.13.2
export BUILD_SYCL_TLA_KERNELS=ON
export BASIC_KERNELS_ENABLED=OFF
export FA2_KERNELS_ENABLED=OFF
export MOE_KERNELS_ENABLED=OFF
export GDN_KERNELS_ENABLED=ON
export MQA_LOGITS_KERNELS_ENABLED=OFF
export MHC_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=ON
export VLLM_XPU_AOT_DEVICES=bmg
export VLLM_XPU_XE2_AOT_DEVICES=bmg
export MAX_JOBS="${MAX_JOBS:-16}"

python setup.py build_ext --inplace
test -f vllm_xpu_kernels/_xpu_C.abi3.so
test -f vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
