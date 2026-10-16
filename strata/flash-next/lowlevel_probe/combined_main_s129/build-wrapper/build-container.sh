#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ $(id -u) == 1000 && ! -e /dev/dri && -w /out ]]
renice 19 -p $$ >/dev/null
ionice -c3 -p $$
export TMPDIR=/out/tmp XDG_CACHE_HOME=/out/cache SYCL_CACHE_DIR=/out/cache
export SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1 NEO_CACHE_DIR=/out/offline-cache NEO_CACHE_STATS=1
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /work/src-ep/build-sycl-ep
sha256sum -c /prep/expected-applied-source.sha256
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-build-system.sha256
cp /prep/expected-applied-source.sha256 /out/applied-source-before.sha256
icpx --version > /out/compiler-version.txt
cmp /out/compiler-version.txt /prep/compiler-version.txt
python3 /prep/guard.py before
mkdir /out/before
for f in strata CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o CMakeFiles/strata.dir/src/program/generate.cpp.o CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o .ninja_log .ninja_deps;do
 cp --parents --reflink=auto "$f" /out/before/
done
sha256sum /out/before/CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o /out/before/CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o /out/before/CMakeFiles/strata.dir/src/program/generate.cpp.o /out/before/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o > /out/objects-before.sha256
cp CMakeCache.txt /out/CMakeCache.before.txt
cp build.ninja /out/build.ninja.before.txt
cp CMakeFiles/rules.ninja /out/rules.ninja.before.txt
python3 /prep/guard.py cache-before
printf '%s\n' 'cmake -S /work/src-ep/sycl -B /work/src-ep/build-sycl-ep -DSTRATA_SYCL_AOT_CACHE_DIR=/out/offline-cache' > /out/configure-command.txt
cmake -S /work/src-ep/sycl -B /work/src-ep/build-sycl-ep -DSTRATA_SYCL_AOT_CACHE_DIR=/out/offline-cache > /out/configure.log 2>&1
python3 /prep/guard.py configured
cp CMakeCache.txt /out/CMakeCache.configured.txt
cp build.ninja /out/build.ninja.configured.txt
cp CMakeFiles/rules.ninja /out/rules.ninja.configured.txt
ninja -n -v strata > /out/ninja-plan.txt
python3 /prep/guard.py plan
TIMEFORMAT='incremental_build_seconds=%R'
{ time ninja -j4 -v strata; } > /out/ninja.log 2> /out/ninja.stderr-time
sha256sum -c /prep/expected-applied-source.sha256
sha256sum -c /prep/expected-unchanged-source.sha256
cmp CMakeCache.txt /out/CMakeCache.configured.txt
cmp build.ninja /out/build.ninja.configured.txt
cmp CMakeFiles/rules.ninja /out/rules.ninja.configured.txt
python3 /prep/guard.py after
python3 /prep/guard.py cache-after
cp --reflink=auto strata /out/strata-s129
cmp strata /out/strata-s129
# Each new main/object/changed archive is hashed once, here; no large prehash loops.
sha256sum /out/strata-s129 CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o CMakeFiles/strata.dir/src/program/generate.cpp.o CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o libstrata_engine.a libstrata_prefill.a libstrata_kernels.a /work/src-ep/sycl/CMakeLists.txt /work/src-ep/sycl/src/core/mtp.cpp /work/src-ep/sycl/include/strata/core/mtp.hpp /work/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp CMakeCache.txt build.ninja CMakeFiles/rules.ninja > /out/build-sha256.txt
cat /out/build-sha256.txt
