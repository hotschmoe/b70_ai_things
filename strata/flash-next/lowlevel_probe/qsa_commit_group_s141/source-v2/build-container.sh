#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
cd /out
sha256sum -c /src/expected-input.sha256 > input-before.txt
icpx --version > compiler-version.txt
SRC=/work/src-ep
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1 -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='host_fixture_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/oracle.cpp -o oracle.o > compile.log 2>&1; } 2> compile.time
QSA=$SRC/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_qsa_indexer.dp.cpp.o
ROPE=$SRC/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_rope.dp.cpp.o
# Prelink only two unchanged existing objects. No main, archive walk or new kernel TU.
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='existing_QSA_rope_AOT_seconds=%R'
{ time icpx "${IMAGES[@]}" -fsycl-link "$QSA" "$ROPE" -o existing.images.o > prelink.log 2>&1; } 2> prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -O3 -DNDEBUG -fp-model=precise oracle.o "$QSA" "$ROPE" existing.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o qsa_three_steps > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-input.sha256 > input-after.txt
sha256sum /src/oracle.cpp oracle.o existing.images.o qsa_three_steps > build-sha256.txt
printf '%s\n' BUILD_COMPLETE_EXISTING_PUBLIC_KERNELS_ONLY > status.txt
