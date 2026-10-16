#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
renice 19 -p $$ >/dev/null
ionice -c 3 -p $$
command -v icpx sha256sum nm renice ionice >/dev/null
[[ -w /out ]]
cd /out
SRC=/work/src-ep
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra -DSTRATA_Q6K_EXPERTS=1 '-DSTRATA_VERSION="0.1.44.1"' -DSTRATA_XMX_SG16=1 -I"$SRC/third_party/ggml" -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/q5_runtime_ones_leaf.cpp -o q5_runtime_ones_leaf.o; } 2> adapter-compile.time
IMAGES=(-O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='adapter_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link q5_runtime_ones_leaf.o -o leaf.images.o > adapter-prelink.log 2>&1; } 2> adapter-prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fp-model=precise q5_runtime_ones_leaf.o leaf.images.o /common/mmvq-core.images.o -Wl,--start-group /frozen/libstrata_kernels.a /frozen/libstrata_core.a -Wl,--end-group /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o q5_runtime_ones_s126 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-container.sha256
nm -C q5_runtime_ones_s126 > symbols.txt
sha256sum /src/q5_runtime_ones_leaf.cpp /src/private_q5_runtime_ones.hpp /common/mmvq-core.images.o /frozen/libstrata_kernels.a /frozen/libstrata_core.a q5_runtime_ones_leaf.o leaf.images.o q5_runtime_ones_s126 > sha256.txt
cat sha256.txt
