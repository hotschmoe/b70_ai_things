#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ $(id -u) == 1000 && ! -e /dev/dri && -w /out/tmp ]]
command -v icpx python3 ar cmp sha256sum readelf renice ionice >/dev/null
renice 19 -p $$ >/dev/null
ionice -c3 -p $$
export TMPDIR=/out/tmp XDG_CACHE_HOME=/out/cache SYCL_CACHE_DIR=/out/cache
export SYCL_CACHE_PERSISTENT=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
cd /out
sha256sum -c /src/expected-container.sha256
icpx --version > compiler-version.txt
mkdir bound-objects
for object in sampler.dp.cpp.o verify_kernels.dp.cpp.o;do
 ar p /work/src-ep/build-sycl-ep/libstrata_kernels.a "$object" > "bound-objects/$object"
 cmp "bound-objects/$object" "/work/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/$object"
done
sha256sum bound-objects/* > original-objects.sha256
# Compile only the private TU (new top1 kernel plus host oracle); no original kernel recompilation.
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen
 -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise
 -I/src -I/work/src-ep/sycl/include -I/work/src-ep/include -I/work/src-ep/third_party/ggml
 -Wall -Wextra -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable)
printf '%q ' icpx "${FLAGS[@]}" -MD -MF oracle.d -c /src/oracle.cpp -o oracle.o > compile-command.txt
printf '\n' >> compile-command.txt
TIMEFORMAT='private_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -MD -MF oracle.d -c /src/oracle.cpp -o oracle.o; } > compile.log 2> compile.stderr-time
B=/work/src-ep/build-sycl-ep
LINK=(-O3 -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel
 -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen
 '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt' -qmkl=sequential
 oracle.o -Wl,--start-group "$B/libstrata_kernels.a" "$B/libstrata_core.a" "$B/libstrata_spec.a"
 "$B/libstrata_kernels_cpu.a" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a"
 -Wl,--end-group -lze_loader -o greedy_top1_s112)
printf '%q ' icpx "${LINK[@]}" > link-command.txt
printf '\n' >> link-command.txt
TIMEFORMAT='private_link_seconds=%R'
{ time icpx "${LINK[@]}"; } > link.log 2> link.stderr-time
sha256sum -c /src/expected-container.sha256
sha256sum /src/oracle.cpp /src/private_argmax.hpp oracle.o greedy_top1_s112 > build-sha256.txt
readelf -sW oracle.o > private-object-symbols.txt
readelf -SW greedy_top1_s112 > executable-sections.txt
readelf -sW greedy_top1_s112 > executable-symbols.txt
printf '%s\n' BUILD_PASS_IMAGE_BINDING_AUDIT_REQUIRED_BEFORE_GPU > status.txt
