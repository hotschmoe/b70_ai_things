#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
renice 19 -p $$ >/dev/null
ionice -c 3 -p $$
cd /out
SRC=/work/src-ep
INPUT=/src/frozen-inputs
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
COMMON=(
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/src/core/device.dp.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/src/core/graph.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/src/core/pinned.dp.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/src/core/weights.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/work/src-ep/src/core/layout.cpp.o"
 "/work/src-ep/build-sycl-ep/CMakeFiles/strata_core.dir/work/src-ep/src/platform/memory.cpp.o"
)
sha256sum "${COMMON[@]}" /work/src-ep/build-sycl-ep/libstrata_kernels.a /work/src-ep/build-sycl-ep/libstrata_core.a > common-inputs.sha256
cmp "$INPUT/libstrata_kernels.a" /work/src-ep/build-sycl-ep/libstrata_kernels.a
cmp "$INPUT/libstrata_core.a" /work/src-ep/build-sycl-ep/libstrata_core.a
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='common_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link "${COMMON[@]}" -o mmvq-core.images.o > common-prelink.log 2>&1; } 2> common-prelink.time
sha256sum -c common-inputs.sha256
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra -DSTRATA_Q6K_EXPERTS=1 '-DSTRATA_VERSION="0.1.44.1"' -DSTRATA_XMX_SG16=1 -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/iq4xs_rpw_leaf.cpp -o iq4xs_rpw_leaf.o; } 2> adapter-compile.time

TIMEFORMAT='adapter_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link iq4xs_rpw_leaf.o -o leaf.images.o > adapter-prelink.log 2>&1; } 2> adapter-prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG iq4xs_rpw_leaf.o leaf.images.o mmvq-core.images.o -Wl,--start-group "$INPUT/libstrata_kernels.a" "$INPUT/libstrata_core.a" -Wl,--end-group /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o iq4xs_rpw_leaf_s59 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-container.sha256
sha256sum -c common-inputs.sha256
nm -C iq4xs_rpw_leaf_s59 > symbols.txt
sha256sum /src/iq4xs_rpw_leaf.cpp "$INPUT/libstrata_kernels.a" "$INPUT/libstrata_core.a" iq4xs_rpw_leaf.o iq4xs_rpw_leaf_s59 mmvq-core.images.o leaf.images.o > sha256.txt
cat sha256.txt
