#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(id -u) == 1000 ]]
cd /out
sha256sum -c /src/expected-input.sha256 > input-before.txt
icpx --version > compiler-version.txt
SRC=/work/src-ep
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1 -I/src -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='host_fixture_and_two_private_kernels_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/oracle.cpp -o oracle.o > compile.log 2>&1; } 2> compile.time
PUBLIC=$SRC/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.dp.cpp.o
# PUBLIC baseline stays the existing object, not a recompiled private clone.
# Only this 503792B public object plus the tiny two-kernel private leaf are linked.
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='one_public_object_plus_two_private_kernels_AOT_seconds=%R'
{ time icpx "${IMAGES[@]}" -fsycl-link "$PUBLIC" oracle.o -o public-private.images.o > prelink.log 2>&1; } 2> prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -O3 -DNDEBUG -fp-model=precise oracle.o "$PUBLIC" public-private.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o gdn_layers_s145 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-input.sha256 > input-after.txt
sha256sum /src/oracle.cpp /src/private_gdn.hpp oracle.o public-private.images.o gdn_layers_s145 > build-sha256.txt
printf '%s\n' BUILD_COMPLETE_PUBLIC_CONTROL_TWO_PRIVATE_KERNELS_ONLY > status.txt
