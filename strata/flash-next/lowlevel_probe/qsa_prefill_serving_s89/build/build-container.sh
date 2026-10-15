#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ $(id -u) == 1000 && ! -e /dev/dri && -w /out ]]
[[ ${S89_EXPECTED_QSA_SHA256:?root-reviewed candidate SHA required} =~ ^[0-9a-f]{64}$ ]]
command -v icpx ninja sha256sum stat cp cmp python3 renice ionice >/dev/null
renice 19 -p $$ >/dev/null
ionice -c3 -p $$
export TMPDIR=/out/tmp XDG_CACHE_HOME=/out/cache SYCL_CACHE_DIR=/out/cache
export SYCL_CACHE_PERSISTENT=0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /work/src-ep/build-sycl-ep
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-s87-build-inputs.sha256
printf '%s  %s\n' "$S89_EXPECTED_QSA_SHA256" /work/src-ep/sycl/src/kernels/cuda/qsa_select.dp.cpp > /out/applied-qsa-source.sha256
sha256sum -c /out/applied-qsa-source.sha256
python3 - <<'PYOWN'
from pathlib import Path
import json,os
names=['.ninja_log','.ninja_deps','strata','libstrata_kernels.a','CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.dp.cpp.o'];d={}
for n in names:
 p=Path(n);s=p.stat();d[n]={'uid':s.st_uid,'gid':s.st_gid,'mode':oct(s.st_mode&511)};assert s.st_uid==1000 and os.access(p,os.W_OK)and os.access(p.parent,os.W_OK),n
Path('/out/actual-ownership.json').write_text(json.dumps(d,indent=2)+'\n')
PYOWN
grep -Fx 'CMAKE_EXE_LINKER_FLAGS:STRING=-fsycl-max-parallel-link-jobs=4' CMakeCache.txt
icpx --version > /out/compiler-version.txt
cp CMakeCache.txt /out/CMakeCache.before.txt
cp build.ninja /out/build.ninja.before.txt
sha256sum libstrata_engine.a libstrata_core.a libstrata_prefill.a libstrata_spec.a libstrata_kernels_cpu.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o CMakeFiles/strata_engine.dir/src/core/verify.cpp.o CMakeFiles/strata.dir/src/program/generate.cpp.o CMakeFiles/strata_kernels.dir/src/kernels/cuda/fused_gr.dp.cpp.o > /out/unchanged-inputs.sha256
mkdir /out/before
for f in strata libstrata_kernels.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.dp.cpp.o .ninja_log .ninja_deps;do cp --parents --reflink=auto "$f" /out/before/;done
find /out/before -type f -exec sha256sum {} + > /out/before.sha256
ninja -n -v strata > /out/ninja-plan.txt
python3 - <<'PYPLAN'
from pathlib import Path
import re,shlex
compiled=[];archives=[];links=0
for l in Path('/out/ninja-plan.txt').read_text().splitlines():
 if not re.match(r'^\[\d+/\d+\] ',l):continue
 cmd=l.split('] ',1)[1];a=shlex.split(cmd);assert '--regenerate-during-build'not in l,'unexpected configure'
 if ' -c 'in cmd:compiled.append(a[a.index('-c')+1])
 elif'/ar 'in cmd or'llvm-ar 'in cmd:assert 'libstrata_kernels.a'in a,cmd;archives.append('libstrata_kernels.a')
 else:assert '-o'in a and a[a.index('-o')+1]=='strata',cmd;links+=1
assert compiled==['/work/src-ep/sycl/src/kernels/cuda/qsa_select.dp.cpp'],compiled
assert archives==['libstrata_kernels.a']and links==1,(archives,links)
PYPLAN
TIMEFORMAT='incremental_build_seconds=%R'
{ time ninja -j4 -v strata; } > /out/ninja.log 2> /out/ninja.stderr-time
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-s87-build-inputs.sha256
sha256sum -c /out/applied-qsa-source.sha256
sha256sum -c /out/unchanged-inputs.sha256
cmp CMakeCache.txt /out/CMakeCache.before.txt
cmp build.ninja /out/build.ninja.before.txt
cp --reflink=auto strata /out/strata-qsa-prefill-s89
cmp strata /out/strata-qsa-prefill-s89
sha256sum /out/strata-qsa-prefill-s89 libstrata_kernels.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.dp.cpp.o libstrata_engine.a CMakeCache.txt > /out/build-sha256.txt
cat /out/build-sha256.txt
