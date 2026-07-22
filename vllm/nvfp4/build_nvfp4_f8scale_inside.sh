#!/usr/bin/env bash
set -euo pipefail

cd /build/vllm-xpu-kernels
pip install --no-build-isolation -q setuptools_scm 2>/dev/null || true
export VLLM_VERSION_OVERRIDE=0.1.10
export BASIC_KERNELS_ENABLED=OFF
export FA2_KERNELS_ENABLED=OFF
export MOE_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=ON
export GDN_KERNELS_ENABLED=ON
export BUILD_SYCL_TLA_KERNELS=ON
export VLLM_XPU_AOT_DEVICES=bmg
export VLLM_XPU_XE2_AOT_DEVICES=bmg
export MAX_JOBS="${MAX_JOBS:-16}"
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu

rm -rf build .deps vllm_xpu_kernels/_xpu_C.abi3.so \
  vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
python setup.py build_ext --inplace
test -f vllm_xpu_kernels/_xpu_C.abi3.so
test -f vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
