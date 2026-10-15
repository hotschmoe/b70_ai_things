#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd /out
sha256sum -c /src/expected-source.sha256
icpx --version > compiler-version.txt
SRC=/work/src-ep
ar p "$SRC/build-sycl-ep/libstrata_kernels.a" qsa_decode_attn.dp.cpp.o > public-current-member.o
cmp public-current-member.o /src/frozen/qsa_decode_attn.dp.cpp.o
sha256sum public-current-member.o > public-current-member.sha256
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -I"$SRC/sycl/include" -I"$SRC/include" -I"$SRC/third_party/ggml")
IMAGES=(-O3 -DNDEBUG -fp-model=precise -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt')
TIMEFORMAT='public_attn_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link /src/frozen/qsa_decode_attn.dp.cpp.o -o public.images.o > public-prelink.log 2>&1; } 2> public-prelink.time
TIMEFORMAT='private_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -Dqsa_decode_attn_batch=reduce12_batch -Dqsa_decode_attn_step=reduce12_step -Dqsa_decode_attn_scratch_floats=reduce12_scratch -c /src/qsa-score-reduce12.cpp -o candidate.o; } 2> private-compile.time
TIMEFORMAT='fixture_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/attn_score_probe.cpp -o probe.o; } 2> fixture-compile.time
TIMEFORMAT='private_images_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link candidate.o probe.o -o private.images.o > private-prelink.log 2>&1; } 2> private-prelink.time
TIMEFORMAT='ordinary_host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fp-model=precise probe.o candidate.o private.images.o /src/frozen/qsa_decode_attn.dp.cpp.o public.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o attn_score_probe > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-source.sha256
ar p "$SRC/build-sycl-ep/libstrata_kernels.a" qsa_decode_attn.dp.cpp.o > public-after.o
cmp public-after.o public-current-member.o
nm -C --defined-only attn_score_probe > symbols.txt
sha256sum attn_score_probe probe.o candidate.o public.images.o private.images.o /src/frozen/qsa_decode_attn.dp.cpp.o > sha256.txt
