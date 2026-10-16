#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(id -u) == 1000 ]]
cd /out
sha256sum -c /src/expected-input.sha256 > input-before.txt
icpx --version > compiler-version.txt
SRC=/work/src-ep
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1 -I/src -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='S146_host_fixture_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/oracle.cpp -o oracle.o > compile.log 2>&1; } 2> compile.time
PUBLIC=$SRC/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.dp.cpp.o
# No -fsycl-link/AOT: retain actual S145 compiled public/private images unchanged.
# New tag helper records a host command; no new device kernel is introduced.
TIMEFORMAT='S146_host_link_reusing_S145_device_images_seconds=%R'
{ time icpx -O3 -DNDEBUG -fp-model=precise oracle.o "$PUBLIC" /prior/public-private.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o gdn_native_tags_s146 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-input.sha256 > input-after.txt
sha256sum /src/oracle.cpp /src/private_gdn.hpp oracle.o gdn_native_tags_s146 > build-sha256.txt
printf '%s\n' BUILD_COMPLETE_HOST_ADAPTER_REUSED_S145_DEVICE_IMAGES > status.txt
