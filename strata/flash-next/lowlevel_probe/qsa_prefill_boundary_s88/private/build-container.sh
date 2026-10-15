#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
renice 19 -p $$ >/dev/null; ionice -c 3 -p $$
cd /out
sha256sum -c /src/expected-source.sha256
icpx --version > compiler-version.txt
# Exact existing archive member must equal source-prepared standalone object.
# No archive hash/copy, main TU compile, or common native AOT regeneration.
ar p /work/src-ep/build-sycl-ep/libstrata_kernels.a qsa_select.dp.cpp.o > qsa-from-archive.o
cmp qsa-from-archive.o /src/frozen/qsa_select.dp.cpp.o
sha256sum qsa-from-archive.o > public-member.sha256
nm -C --defined-only qsa-from-archive.o > public-member-symbols.txt
SRC=/work/src-ep
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -I"$SRC/sycl/include" -I"$SRC/include")
IMAGES=(-O3 -DNDEBUG -fp-model=precise -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
# Reuse the exact previously audited PUBLIC wrapper; no public AOT invocation.
sha256sum /public/qsa.images.o > reused-public-wrapper.sha256
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/scorer_leaf.cpp -o scorer_leaf.o; } 2> adapter-compile.time
TIMEFORMAT='adapter_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link scorer_leaf.o -o leaf.images.o > adapter-prelink.log 2>&1; } 2> adapter-prelink.time
# Host CASES/counts only: require the private device-image wrapper unchanged.
cmp leaf.images.o /public/leaf.images.o
sha256sum leaf.images.o /public/leaf.images.o > unchanged-private-wrapper.sha256
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fp-model=precise scorer_leaf.o leaf.images.o /src/frozen/qsa_select.dp.cpp.o /public/qsa.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o scorer_leaf_s88 > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-source.sha256
cmp qsa-from-archive.o /src/frozen/qsa_select.dp.cpp.o
nm -C scorer_leaf_s88 > symbols.txt
sha256sum /src/scorer_leaf.cpp /src/private_tiled_scores.hpp /src/frozen/qsa_select.dp.cpp.o scorer_leaf.o /public/qsa.images.o leaf.images.o scorer_leaf_s88 > sha256.txt
