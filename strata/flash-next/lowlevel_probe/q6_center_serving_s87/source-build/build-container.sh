#!/usr/bin/env bash
# ROOT ONLY after reviewed exact native apply. No configure or GPU.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ $(id -u) == 1000 && ! -e /dev/dri && -w /out ]]
command -v icpx ninja sha256sum stat cp cmp python3 renice ionice >/dev/null
renice 19 -p $$ >/dev/null
ionice -c3 -p $$
export TMPDIR=/out/tmp XDG_CACHE_HOME=/out/cache SYCL_CACHE_DIR=/out/cache
export SYCL_CACHE_PERSISTENT=0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /work/src-ep/build-sycl-ep
sha256sum -c /prep/expected-build-source.sha256
python3 - <<'PYOWN'
from pathlib import Path
import json,os
names=['.ninja_log','.ninja_deps','strata','libstrata_kernels.a','libstrata_engine.a','CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o','CMakeFiles/strata_engine.dir/src/core/verify.cpp.o','CMakeFiles/strata.dir/src/program/generate.cpp.o']
d={}
for n in names:
 p=Path(n);s=p.stat();d[n]={'uid':s.st_uid,'gid':s.st_gid,'mode':oct(s.st_mode&511)}
 assert s.st_uid==1000 and os.access(p,os.W_OK)and os.access(p.parent,os.W_OK),n
Path('/out/actual-write-permissions.json').write_text(json.dumps(d,indent=2)+'\n')
PYOWN
# Exact existing ABI/config, no configure; four AOT jobs already fixed in cache.
grep -Fx 'CMAKE_EXE_LINKER_FLAGS:STRING=-fsycl-max-parallel-link-jobs=4' CMakeCache.txt
icpx --version > /out/compiler-version.txt
cp CMakeCache.txt /out/CMakeCache.before.txt
cp build.ninja /out/build.ninja.before.txt
sha256sum libstrata_core.a libstrata_prefill.a libstrata_spec.a libstrata_kernels_cpu.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/fused_gr.dp.cpp.o > /out/unchanged-inputs.sha256
# Preserve actual prebuild artifacts before any Ninja mutation; stale03bb is labeled.
mkdir /out/before
for p in strata libstrata_kernels.a libstrata_engine.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o CMakeFiles/strata_engine.dir/src/core/verify.cpp.o CMakeFiles/strata.dir/src/program/generate.cpp.o .ninja_log .ninja_deps;do
 cp --parents --reflink=auto "$p" /out/before/
done
find /out/before -type f -exec sha256sum {} + > /out/before.sha256
ninja -n -v strata > /out/ninja-plan.txt
python3 - <<'PYPLAN'
from pathlib import Path
import re,shlex
expected={'/work/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp','/work/src-ep/sycl/src/core/verify.cpp','/work/src-ep/sycl/src/program/generate.cpp'}
compiled=[];archives=[];links=0
for line in Path('/out/ninja-plan.txt').read_text().splitlines():
 if not re.match(r'^\[\d+/\d+\] ',line):continue
 cmd=line.split('] ',1)[1];a=shlex.split(cmd);assert '--regenerate-during-build'not in line,'Unexpected configure'
 if ' -c 'in cmd:compiled.append(a[a.index('-c')+1])
 elif('/ar 'in cmd or'llvm-ar 'in cmd):
  names=[n for n in ('libstrata_kernels.a','libstrata_engine.a')if n in a];assert len(names)==1,cmd;archives+=names
 else:assert '-o'in a and a[a.index('-o')+1]=='strata',cmd;links+=1
assert len(compiled)==3 and set(compiled)==expected,compiled
assert sorted(archives)==['libstrata_engine.a','libstrata_kernels.a']and links==1,(archives,links)
PYPLAN
TIMEFORMAT='incremental_build_seconds=%R'
{ time ninja -j4 -v strata; } > /out/ninja.log 2> /out/ninja.stderr-time
sha256sum -c /prep/expected-build-source.sha256
sha256sum -c /out/unchanged-inputs.sha256
cmp CMakeCache.txt /out/CMakeCache.before.txt
cmp build.ninja /out/build.ninja.before.txt
cp --reflink=auto strata /out/strata-q6-center-s87
cmp strata /out/strata-q6-center-s87
sha256sum /out/strata-q6-center-s87 libstrata_kernels.a libstrata_engine.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o CMakeFiles/strata_engine.dir/src/core/verify.cpp.o CMakeFiles/strata.dir/src/program/generate.cpp.o CMakeCache.txt > /out/build-sha256.txt
cat /out/build-sha256.txt
