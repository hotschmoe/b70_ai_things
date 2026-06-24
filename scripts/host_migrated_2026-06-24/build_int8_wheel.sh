#!/bin/bash
set -uo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export CCACHE_DIR=/mnt/vm_8tb/b70/.ccache TMPDIR=/mnt/vm_8tb/b70/.tmp PIP_CACHE_DIR=/mnt/vm_8tb/b70/.pipcache
export MAX_JOBS=24
export FA2_KERNELS_ENABLED=OFF MOE_KERNELS_ENABLED=OFF GDN_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF BASIC_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF XPU_SPECIFIC_KERNELS_ENABLED=ON
cd /src
echo "=== WHEEL START $(date) ==="
pip wheel --no-build-isolation --no-deps -w /src/int8_wheel_dist .
echo "=== WHEEL_RC=$? $(date) ==="
ls -la /src/int8_wheel_dist
