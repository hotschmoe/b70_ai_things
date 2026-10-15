#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
renice 19 -p $$ >/dev/null
ionice -c 3 -p $$
[[ $(id -u) == 1000 && ! -e /dev/dri ]]
cd /out
SRC=/work/src-ep
sha256sum -c /src/expected-container.sha256
python3 - <<'PYSTAT'
from pathlib import Path
import json
r=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
m={'/common/mmvq-core.images.o':r/'iq4xs-rpw2-s59-prep/build/mmvq-core.images.o','/frozen/libstrata_kernels.a':r/'iq4xs-rpw2-s59-prep/frozen-inputs/libstrata_kernels.a','/frozen/libstrata_core.a':r/'iq4xs-rpw2-s59-prep/frozen-inputs/libstrata_core.a','/work/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o':Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o')}
d={str(host):{k:getattr(Path(n).stat(),k)for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}for n,host in m.items()}
Path('/out/immutable-input-stats.json').write_text(json.dumps(d,indent=2)+'\n')
PYSTAT
icpx --version > compiler-version.txt
FLAGS=(-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra -DSTRATA_Q6K_EXPERTS=1 '-DSTRATA_VERSION="0.1.44.1"' -DSTRATA_XMX_SG16=1 -I"$SRC/third_party/ggml" -I"$SRC/sycl/include" -I"$SRC/include")
TIMEFORMAT='adapter_compile_seconds=%R'
{ time icpx "${FLAGS[@]}" -c /src/routed_iq2_leaf.cpp -o routed_iq2_leaf.o; } 2> adapter-compile.time
IMAGES=(-O3 -DNDEBUG -fp-model=precise -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt")
TIMEFORMAT='adapter_prelink_seconds=%R'
{ time icpx -v "${IMAGES[@]}" -fsycl-link routed_iq2_leaf.o -o leaf.images.o > adapter-prelink.log 2>&1; } 2> adapter-prelink.time
TIMEFORMAT='host_link_seconds=%R'
{ time icpx -v -O3 -DNDEBUG routed_iq2_leaf.o leaf.images.o /common/mmvq-core.images.o -Wl,--start-group /frozen/libstrata_kernels.a /frozen/libstrata_core.a -Wl,--end-group /usr/lib/x86_64-linux-gnu/libze_loader.so -lsycl -lsycl-devicelib-host -Wl,-Map=/out/link.map -o routed_iq2_leaf_s98 > host-link.log 2>&1; } 2> host-link.time
awk '$2 ~ /^\/src\// || $2 ~ /iq_kernels.dp.cpp$/ {print}' /src/expected-container.sha256 | sha256sum -c -
nm -C routed_iq2_leaf_s98 > symbols.txt
sha256sum /src/routed_iq2_leaf.cpp /src/private_iq2_gu.hpp routed_iq2_leaf.o leaf.images.o routed_iq2_leaf_s98 > sha256.txt
awk '$2 ~ /^\/(common|frozen)\// {print}' /src/expected-container.sha256 >> sha256.txt
cat sha256.txt
