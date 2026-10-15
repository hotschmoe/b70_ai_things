#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd /out
SRC=/work/src-ep
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
ar p /frozen/libstrata_kernels.a fused_gr.dp.cpp.o > hc-fat.o
ar p /frozen/libstrata_kernels.a verify_kernels.dp.cpp.o > verify-support-fat.o
printf '%s\n' '27d0ab6aff30126bd317584b4b2cbacd0b4667764f9cf624738625c787c07d99  hc-fat.o' '584e694fc1ece4740fe4550e2c34fafd28a77921d81960039d01f3c27dbb6613  verify-support-fat.o' | sha256sum -c -
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1 -I"$SRC/sycl/include" -I"$SRC/include")
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='HC_support_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link hc-fat.o verify-support-fat.o -o hc-support.images.o > hc-prelink.log 2>&1; } 2> hc-prelink.time
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/hc_more_oracle.cpp -o hc_more_oracle.o; } 2> adapter-compile.time
TIMEFORMAT='adapter_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link hc_more_oracle.o -o adapter.images.o > adapter-prelink.log 2>&1; } 2> adapter-prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG hc_more_oracle.o adapter.images.o hc-support.images.o /common/mmvq-core.images.o -Wl,--start-group /frozen/libstrata_kernels.a /frozen/libstrata_core.a -Wl,--end-group /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o hc_more_oracle > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-container.sha256
nm -C hc_more_oracle > symbols.txt
sha256sum /src/hc_more_oracle.cpp hc_more_oracle.o hc_more_oracle hc-fat.o verify-support-fat.o hc-support.images.o adapter.images.o /common/mmvq-core.images.o /frozen/libstrata_kernels.a /frozen/libstrata_core.a > sha256.txt
