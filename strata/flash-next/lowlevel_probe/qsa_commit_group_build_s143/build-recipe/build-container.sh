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
sha256sum -c /prep/expected-build-state.sha256
cp /prep/expected-applied-source.sha256 /out/applied-source-before.sha256
icpx --version > /out/compiler-version.txt
cmp /out/compiler-version.txt /prep/compiler-version.txt
cp /prep/prior-configured-guards.json /out/configured-guards-reused.json
python3 /prep/guard.py before
# NO configure, old main backup/hash, model scan, image audit or cache warm.
python3 /prep/guard.py cache-before
ninja -n -v strata > /out/ninja-plan.txt
python3 /prep/guard.py plan
TIMEFORMAT='incremental_build_seconds=%R'
{ time ninja -j4 -v strata; } > /out/ninja.log 2> /out/ninja.stderr-time
sha256sum -c /prep/expected-applied-source.sha256
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-build-state.sha256
python3 /prep/guard.py cache-after
cp --reflink=auto strata /out/strata-s143
# Exactly the new main/verify object/engine archive hash once, plus tiny bindings.
sha256sum /out/strata-s143 CMakeFiles/strata_engine.dir/src/core/verify.cpp.o libstrata_engine.a /work/src-ep/sycl/CMakeLists.txt /work/src-ep/sycl/src/core/mtp.cpp /work/src-ep/sycl/include/strata/core/mtp.hpp /work/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp /work/src-ep/sycl/src/core/verify.cpp CMakeCache.txt build.ninja CMakeFiles/rules.ninja ggml/src/ggml-version.h > /out/build-sha256.txt
cat /out/build-sha256.txt
