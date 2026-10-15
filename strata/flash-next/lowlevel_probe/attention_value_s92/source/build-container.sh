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
# Rename ALL 11 class tags, including inactive modes/step/merge and lane-cell.
# Function renaming alone did not isolate weak host kernel metadata in S13/S91.
RENAMES=(
 -Dattn_chunk_kernel_6e87d5=s92_private_attn_chunk_kernel_6e87d5
 -Dattn_chunk_kernel_dfacfc=s92_private_attn_chunk_kernel_dfacfc
 -Dattn_chunk_kernel_b0d448=s92_private_attn_chunk_kernel_b0d448
 -Dattn_chunk_kernel_2ad0a4=s92_private_attn_chunk_kernel_2ad0a4
 -Dattn_chunk_kernel_5c04f7=s92_private_attn_chunk_kernel_5c04f7
 -Dattn_merge_kernel_338175=s92_private_attn_merge_kernel_338175
 -Dattn_chunk_kernel_f262e6=s92_private_attn_chunk_kernel_f262e6
 -Dattn_chunk_kernel_4fd923=s92_private_attn_chunk_kernel_4fd923
 -Dattn_chunk_kernel_f6855e=s92_private_attn_chunk_kernel_f6855e
 -Dattn_chunk_kernel_7c8e09=s92_private_attn_chunk_kernel_7c8e09
 -Dattn_merge_kernel_c055e7=s92_private_attn_merge_kernel_c055e7
)
TIMEFORMAT='private_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" "${RENAMES[@]}" -Dqsa_decode_attn_batch=value_batch -Dqsa_decode_attn_step=value_step -Dqsa_decode_attn_scratch_floats=value_scratch -c /src/qsa-value-gather.cpp -o candidate.o; } 2> private-compile.time
TIMEFORMAT='fixture_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/attn_value_probe.cpp -o probe.o; } 2> fixture-compile.time
TIMEFORMAT='private_images_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link candidate.o probe.o -o private.images.o > private-prelink.log 2>&1; } 2> private-prelink.time
TIMEFORMAT='ordinary_host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG -fp-model=precise probe.o candidate.o private.images.o /src/frozen/qsa_decode_attn.dp.cpp.o /src/frozen/public.images.o /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o attn_value_probe > host-link.log 2>&1; } 2> host-link.time
sha256sum -c /src/expected-source.sha256
ar p "$SRC/build-sycl-ep/libstrata_kernels.a" qsa_decode_attn.dp.cpp.o > public-after.o
cmp public-after.o public-current-member.o
nm -C --defined-only candidate.o > candidate.symbols.txt
nm -C --defined-only /src/frozen/qsa_decode_attn.dp.cpp.o > public.symbols.txt
nm -C --defined-only attn_value_probe > symbols.txt
objdump -drC candidate.o > candidate.host-assembly.txt
objdump -drC /src/frozen/qsa_decode_attn.dp.cpp.o > public.host-assembly.txt
sha256sum attn_value_probe probe.o candidate.o private.images.o /src/frozen/public.images.o /src/frozen/qsa_decode_attn.dp.cpp.o > sha256.txt
