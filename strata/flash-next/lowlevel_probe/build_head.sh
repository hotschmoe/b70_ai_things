#!/usr/bin/env bash
# CPU-only build of head_cols_probe (S25 revision) in strata-ab-build:ocloc (--network none, no /dev/dri) against the
# FROZEN build-sycl-ep archives, with the exact flags of build-sycl-ep/build.ninja.  Headers from src-ep (read only).
# Objects/binary stay in this folder.  Run from the host:  bash build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build/head-cols"
mkdir -p "$OUT"
docker run --rm --network none -v /mnt/vm_8tb/b70/build/strata-ab:/work -v "$OUT":/out -v "$HERE":/src:ro --entrypoint bash strata-ab-build:ocloc -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd /out
SRC=/work/src-ep
BLD=/work/src-ep/build-sycl-ep
DEFS="-DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERSION=\"0.1.44.1\" -DSTRATA_XMX_SG16=1"
FLAGS="-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra"
INC="-I$SRC/sycl/include -I$SRC/include"
LINK="-fsycl-max-parallel-link-jobs=8 -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen"
icpx --version | head -1
set -x
icpx $DEFS $FLAGS $INC -c /src/head_cols_probe.cpp -o head_cols_probe.o
icpx -O3 -DNDEBUG $LINK "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" -fsycl-device-code-split=per_kernel \
     head_cols_probe.o $BLD/libstrata_kernels.a $BLD/libstrata_core.a /usr/lib/x86_64-linux-gnu/libze_loader.so -o head_cols_probe
set +x
sha256sum /src/head_cols_probe.cpp $BLD/libstrata_kernels.a $BLD/libstrata_core.a head_cols_probe.o head_cols_probe > sha256.txt
cat sha256.txt; ls -la head_cols_probe
' 2>&1 | tee "$OUT/build.log"
