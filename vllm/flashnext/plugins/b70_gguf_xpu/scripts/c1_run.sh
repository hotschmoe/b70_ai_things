#!/usr/bin/env bash
# Correctness plus fused-layer bench. Invoke only via bin/gpu-run --card 0.
set -euo pipefail
export ZE_AFFINITY_MASK=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:$ONEAPI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-/mnt/vm_8tb/b70/build/gguf-xpu-sycl-cache}"
"$PY" "$ROOT/tests/gpu_check.py"
"$PY" "$ROOT/tests/moe_layer_bench.py"
