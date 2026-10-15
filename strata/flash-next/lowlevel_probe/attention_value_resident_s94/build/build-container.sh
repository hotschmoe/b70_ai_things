#!/usr/bin/env bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ $(id -u) == 1000 && ! -e /dev/dri && -w /out ]]
command -v icpx ninja sha256sum stat cp cmp python3 renice ionice >/dev/null
renice 19 -p $$ >/dev/null
ionice -c3 -p $$
export TMPDIR=/out/tmp XDG_CACHE_HOME=/out/cache SYCL_CACHE_DIR=/out/cache
export SYCL_CACHE_PERSISTENT=0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /work/src-ep/build-sycl-ep
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-applied-source.sha256
sha256sum -c /prep/expected-before-build.sha256
sha256sum -c /prep/expected-unchanged-build-inputs.sha256
python3 - <<'PYOWN'
from pathlib import Path
import json,os
names=['.ninja_log','.ninja_deps','strata','libstrata_prefill.a','CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o']
d={};bad=[]
for n in names:
 p=Path(n);s=p.stat();d[n]={'uid':s.st_uid,'gid':s.st_gid,'mode':oct(s.st_mode&511),'writable':os.access(p,os.W_OK),'parent_writable':os.access(p.parent,os.W_OK)}
 if s.st_uid!=1000 or not os.access(p,os.W_OK) or not os.access(p.parent,os.W_OK):bad.append(n)
Path('/out/actual-ownership.json').write_text(json.dumps(d,indent=2)+'\n')
assert not bad,('root must review/repair ONLY planned targets before build',bad)
PYOWN
grep -Fx 'CMAKE_EXE_LINKER_FLAGS:STRING=-fsycl-max-parallel-link-jobs=4' CMakeCache.txt
icpx --version > /out/compiler-version.txt
cp CMakeCache.txt /out/CMakeCache.before.txt
cp build.ninja /out/build.ninja.before.txt
mkdir /out/before
# Small rollback/attribution snapshots only. Old main551cca remains frozen as S93.
for f in CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o .ninja_log .ninja_deps;do cp --parents --reflink=auto "$f" /out/before/;done
find /out/before -type f -exec sha256sum {} + > /out/before.sha256
ninja -n -v strata > /out/ninja-plan.txt
python3 - <<'PYPLAN'
from pathlib import Path
import re,shlex
expected={'/work/src-ep/sycl/src/prefill/prefill.cpp'}
compiled=[];archives=[];links=0
for l in Path('/out/ninja-plan.txt').read_text().splitlines():
 if not re.match(r'^\[\d+/\d+\] ',l):continue
 cmd=l.split('] ',1)[1];a=shlex.split(cmd);assert '--regenerate-during-build' not in l,'unexpected configure'
 if ' -c ' in cmd:
  assert '-fp-model=precise' in a,cmd
  compiled.append(a[a.index('-c')+1])
 elif '/ar ' in cmd or 'llvm-ar ' in cmd:
  hits=[x for x in ['libstrata_prefill.a'] if x in a];assert len(hits)==1,cmd;archives+=hits
 else:
  assert '-o' in a and a[a.index('-o')+1]=='strata',cmd
  assert '-fsycl-max-parallel-link-jobs=4' in a and '-fsycl' in a,cmd
  links+=1
assert len(compiled)==1 and set(compiled)==expected,compiled
assert archives==['libstrata_prefill.a'] and links==1,(archives,links)
PYPLAN
TIMEFORMAT='incremental_build_seconds=%R'
{ time ninja -j4 -v strata; } > /out/ninja.log 2> /out/ninja.stderr-time
sha256sum -c /prep/expected-unchanged-source.sha256
sha256sum -c /prep/expected-applied-source.sha256
sha256sum -c /prep/expected-unchanged-build-inputs.sha256
# Existing S93 inputs except the intentional prefill object/archive/main remain fixed.
sha256sum CMakeCache.txt build.ninja > /out/build-system-after.sha256
cmp CMakeCache.txt /out/CMakeCache.before.txt
cmp build.ninja /out/build.ninja.before.txt
cp --reflink=auto strata /out/strata-attn-value-resident-s94
cmp strata /out/strata-attn-value-resident-s94
sha256sum /out/strata-attn-value-resident-s94 libstrata_kernels.a libstrata_prefill.a CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_decode_attn.dp.cpp.o CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o libstrata_engine.a CMakeCache.txt > /out/build-sha256.txt
cat /out/build-sha256.txt
