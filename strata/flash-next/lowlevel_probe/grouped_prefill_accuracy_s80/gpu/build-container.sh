#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
renice 19 -p $$ >/dev/null
ionice -c 3 -p $$
cd /out
SRC=/work/src-ep
BLD=$SRC/build-sycl-ep
sha256sum -c /src/expected-source-container.sha256
command -v icpx sha256sum nm >/dev/null
icpx --version > compiler-version.txt
INPUTS=(/frozen/libstrata_prefill.a /frozen/libstrata_engine.a /frozen/libstrata_spec.a /frozen/libstrata_kernels.a /frozen/libstrata_core.a /frozen/libstrata_kernels_cpu.a /frozen/libggml-cpu.a /frozen/libggml-base.a)
sha256sum "${INPUTS[@]}" > archives-before.sha256
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_XMX_SG16=1 -I$SRC/sycl/include -I$SRC/include -I/work/llama-pinned/ggml/include)
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/grouped_prefill_leaf.cpp -o grouped_prefill_leaf.o; } 2> adapter-compile.time
TIMEFORMAT='adapter_AOT_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fsycl-max-parallel-link-jobs=4 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt' -fsycl-device-code-split=per_kernel -qmkl=sequential grouped_prefill_leaf.o -Wl,--start-group "${INPUTS[@]}" -Wl,--end-group /usr/lib/x86_64-linux-gnu/libze_loader.so -Wl,-Map=/out/link.map -o grouped_prefill_leaf_s80 > AOT-link.log 2>&1; } 2> AOT-link.time
sha256sum "${INPUTS[@]}" > archives-after.sha256
cmp archives-before.sha256 archives-after.sha256
sha256sum -c /src/expected-source-container.sha256
sha256sum /src/grouped_prefill_leaf.cpp /src/reference_bound.hpp grouped_prefill_leaf.o grouped_prefill_leaf_s80 > sha256.txt
nm -C grouped_prefill_leaf_s80 > symbols.txt
