#!/bin/bash
set -uo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export CCACHE_DIR=/mnt/vm_8tb/b70/.ccache
export TMPDIR=/mnt/vm_8tb/b70/.tmp
export PIP_CACHE_DIR=/mnt/vm_8tb/b70/.pipcache
export MAX_JOBS=24
export FA2_KERNELS_ENABLED=OFF MOE_KERNELS_ENABLED=OFF GDN_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF BASIC_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF XPU_SPECIFIC_KERNELS_ENABLED=ON
mkdir -p "$CCACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR"
cd /src
echo "=== START $(date) MINIMAL ==="
pip install --no-build-isolation -e . -v
echo "=== PIP_RC=$? $(date) ==="
