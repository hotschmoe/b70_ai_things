#!/usr/bin/env bash
# CPU-only AOT build of ep_ready_probe (S27) in strata-ab-build:ocloc: --network none, NO /dev/dri, no lease needed.
# Links the FROZEN src-ep/build-sycl-ep archives (mounted read-only); flags copied from lowlevel_probe/build_head.sh.
# All outputs (and compiler temp files) stay in ./build on disk. Run: bash build_ep.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build/ep-ready"
mkdir -p "$OUT/tmp"
docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/out -e TMPDIR=/out/tmp \
  -v /mnt/vm_8tb/b70/build/strata-ab:/work:ro -v "$OUT":/out -v "$HERE":/src:ro --entrypoint bash strata-ab-build:ocloc -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd /out
SRC=/work/src-ep
BLD=/work/src-ep/build-sycl-ep
DEFS="-DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERSION=\"0.1.44.1\" -DSTRATA_XMX_SG16=1"
FLAGS="-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra"
INC="-I$SRC/sycl/include -I$SRC/include"
LINK="-fsycl-max-parallel-link-jobs=8 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen"
icpx --version | head -1
set -x
icpx $DEFS $FLAGS $INC -c /src/ep_ready_probe.cpp -o ep_ready_probe.o
icpx -O3 -DNDEBUG $LINK "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" -fsycl-device-code-split=per_kernel \
     ep_ready_probe.o $BLD/libstrata_kernels.a $BLD/libstrata_core.a /usr/lib/x86_64-linux-gnu/libze_loader.so -o ep_ready_probe
set +x
sha256sum /src/ep_ready_probe.cpp /src/build_ep.sh /src/run_ep.sh $SRC/sycl/include/strata/core/ep_mem.hpp \
  $SRC/sycl/include/strata/kernels/ep_kernels.hpp $SRC/sycl/src/kernels/cuda/ep_kernels.dp.cpp \
  $BLD/libstrata_kernels.a $BLD/libstrata_core.a ep_ready_probe.o ep_ready_probe > sha256.txt
cat sha256.txt; ls -la ep_ready_probe
' 2>&1 | tee "$OUT/build.log"
rmdir "$OUT/tmp" 2>/dev/null || true
