#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(id -u) == 1000 && -w /out ]]
cd /out
sha256sum -c /src/expected-source.sha256
icpx --version > compiler-version.txt
SRC=/work/src-ep
# Confirm the current public archive member without copying/hashing the archive.
ar p "$SRC/build-sycl-ep/libstrata_kernels.a" qsa_decode_attn.dp.cpp.o > public-current-member.o
cmp public-current-member.o /reuse-frozen/qsa_decode_attn.dp.cpp.o
sha256sum public-current-member.o > public-current-member.sha256
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -I"$SRC/sycl/include" -I"$SRC/include" -I"$SRC/third_party/ggml")
IMAGES=(-O3 -DNDEBUG -fp-model=precise -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt')
# No public/private attention source compile and no public AOT regeneration.
TIMEFORMAT='fixture_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/attn_value_probe.cpp -o probe.o; } 2> fixture-compile.time
TIMEFORMAT='private_images_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link /reuse-build/candidate.o probe.o -o private.images.o > private-prelink.log 2>&1; } 2> private-prelink.time
# Wrapper equality is diagnostic only: host metadata/helpers can differ.
if cmp -s private.images.o /reuse-build/private.images.o;then
 printf 'equal\n' > private-wrapper-comparison.txt
else
 printf 'different\n' > private-wrapper-comparison.txt
fi
sha256sum private.images.o /reuse-build/private.images.o > private-wrapper-sha256.txt
# Root's postbuild gate is selected full M1 image equality, independent host
# bindings/registered-name pointers and retained bundles, not wrapper equality.
TIMEFORMAT='ordinary_host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fp-model=precise probe.o /reuse-build/candidate.o private.images.o /reuse-frozen/qsa_decode_attn.dp.cpp.o /reuse-frozen/public.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o attn_value_probe > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-source.sha256
ar p "$SRC/build-sycl-ep/libstrata_kernels.a" qsa_decode_attn.dp.cpp.o > public-after.o
cmp public-after.o public-current-member.o
nm -C --defined-only attn_value_probe > symbols.txt
sha256sum attn_value_probe probe.o private.images.o /reuse-build/candidate.o /reuse-frozen/public.images.o /reuse-frozen/qsa_decode_attn.dp.cpp.o > sha256.txt
