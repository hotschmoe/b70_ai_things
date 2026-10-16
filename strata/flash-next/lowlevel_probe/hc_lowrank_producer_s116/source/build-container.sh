#!/usr/bin/env bash
# Future CPU_BUILD_GO only. Six private kernel classes, no live TU/archive/main.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
cd /out
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen
 -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel
 -fp-model=precise -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1
 -I/src -I/work/src-ep/sycl/include -I/work/src-ep/include)
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen
 -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4
 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='S116_private_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/oracle.cpp -o oracle.o > compile.log 2>&1; } 2> compile.time
TIMEFORMAT='S116_six_HC_AOT_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link oracle.o -o private.images.o > prelink.log 2>&1; } 2> prelink.time
TIMEFORMAT='S116_host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG oracle.o private.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o hc_producer_s116 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-container.sha256
sha256sum /src/oracle.cpp /src/private_hc.hpp /src/anchor_names.hpp oracle.o private.images.o hc_producer_s116 > build-sha256.txt
printf '%s\n' BUILD_COMPLETE_SELECTED_CONTROL_AND_CANDIDATE_IMAGE_AUDIT_REQUIRED_BEFORE_GPU > status.txt
