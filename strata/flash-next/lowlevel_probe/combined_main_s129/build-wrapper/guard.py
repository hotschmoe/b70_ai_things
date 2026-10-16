#!/usr/bin/env python3
# Small S114 plan guard extended to the reviewed S129 eight actions and link-only cache.
import json,os,re,shlex,subprocess,sys
from pathlib import Path
OUT=Path('/out'); PREP=Path('/prep')
COMPILES={
'/work/src-ep/sycl/src/core/mtp.cpp':'CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o',
'/work/src-ep/sycl/src/prefill/prefill.cpp':'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o',
'/work/src-ep/sycl/src/program/generate.cpp':'CMakeFiles/strata.dir/src/program/generate.cpp.o',
'/work/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp':'CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o'}
ARCHIVES={'libstrata_engine.a','libstrata_prefill.a','libstrata_kernels.a'}
CHANGED={'strata',*COMPILES.values(),*ARCHIVES}
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def metadata(p):
 s=Path(p).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,uid=s.st_uid,gid=s.st_gid)
def cachemap(text):
 return dict((a.split(':',1)[0],(a.split(':',1)[1],v)) for x in text.splitlines() if x and not x.startswith(('#','//')) and '=' in x and ':' in (a:=x.split('=',1)[0]) for v in [x.split('=',1)[1]])
def compiles(text):
 return {b.splitlines()[0]:b for b in text.split('\n\n') if b.startswith('build ') and re.search(r': (?:CXX|C)_COMPILER',b.splitlines()[0])}
mode=sys.argv[1]
if mode in ('before','after'):
 baseline=json.loads((PREP/'baseline-stat.json').read_text());actual={p:metadata(p) for p in baseline}
 save('artifact-stat-'+mode+'.json',actual)
 for p,want in baseline.items():
  if mode=='before' or p not in CHANGED: assert actual[p]==want,('artifact continuity',p,want,actual[p])
 if mode=='before':
  for p in CHANGED|{'.ninja_log','.ninja_deps','CMakeCache.txt','build.ninja','CMakeFiles/rules.ninja'}:
   assert Path(p).stat().st_uid==1000 and os.access(p,os.W_OK) and os.access(Path(p).parent,os.W_OK),('ownership/writable',p)
  cache=Path('/out/offline-cache');assert cache.stat().st_uid==1000 and os.access(cache,os.W_OK)
  save('build-environment.json',dict(cwd=os.getcwd(),uid=os.getuid(),gid=os.getgid(),cache=str(cache),cache_uid=cache.stat().st_uid,environment={k:os.environ[k] for k in ('TMPDIR','SYCL_CACHE_PERSISTENT','NEO_CACHE_PERSISTENT','NEO_CACHE_DIR','NEO_CACHE_STATS','MAX_JOBS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
elif mode=='configured':
 before=cachemap((OUT/'CMakeCache.before.txt').read_text());after=cachemap(Path('CMakeCache.txt').read_text())
 assert 'STRATA_SYCL_AOT_CACHE_DIR' not in before
 assert after.pop('STRATA_SYCL_AOT_CACHE_DIR')==('PATH','/out/offline-cache')
 assert before==after,('existing CMake cache changed', {k:(before.get(k),after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)})
 old=compiles((OUT/'build.ninja.before.txt').read_text());new=compiles(Path('build.ninja').read_text())
 assert old and old==new,'compile edges changed (including FLAGS/DEFINES/INCLUDES)'
 assert Path('CMakeFiles/rules.ninja').read_bytes()==(OUT/'rules.ninja.before.txt').read_bytes(),'compiler/link rule templates changed'
 save('configured-guards.json',dict(status='PASS_ALL_EXISTING_CACHE_AND_COMPILE_EDGES_UNCHANGED',compile_edges=len(old),only_new_cache='STRATA_SYCL_AOT_CACHE_DIR:PATH=/out/offline-cache'))
elif mode=='plan':
 compiled={};archives=[];links=[];actions=[]
 for line in (OUT/'ninja-plan.txt').read_text().splitlines():
  if not re.match(r'^\[\d+/\d+\] ',line):continue
  cmd=line.split('] ',1)[1];a=shlex.split(cmd);actions.append(cmd)
  assert '--regenerate-during-build' not in cmd
  if '-c' in a:
   assert a[0]=='/opt/intel/oneapi/compiler/2026.1/bin/icpx',cmd
   assert all(x in a for x in ['-O3','-fp-model=precise','-fsycl','-fsycl-targets=spir64_gen','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel']),cmd
   source=a[a.index('-c')+1];obj=a[a.index('-o')+1];assert source not in compiled
   compiled[source]=obj
  elif '/ar ' in cmd or 'llvm-ar ' in cmd:
   hits=[x for x in ARCHIVES if x in a];assert len(hits)==1,cmd;archives+=hits
  else:
   assert '-o' in a and a[a.index('-o')+1]=='strata',cmd
   assert '/opt/intel/oneapi/compiler/2026.1/bin/icpx' in a,cmd
   assert all(x in a for x in ['-O3','-fsycl-max-parallel-link-jobs=4','-fsycl','-fsycl-targets=spir64_gen','-fsycl-device-code-split=per_kernel']),cmd
   assert a.count('-Xsycl-target-backend=spir64_gen')==1
   assert '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt -allow_caching -cache_dir /out/offline-cache' in a,cmd
   links.append(cmd)
 assert len(actions)==8 and compiled==COMPILES and len(archives)==3 and set(archives)==ARCHIVES and len(links)==1,(len(actions),compiled,archives,links)
 save('root-validated-plan.json',dict(status='PASS_EXACT_EIGHT',compiled=compiled,archives=archives,links=links))
elif mode in ('cache-before','cache-after'):
 cmd=['/usr/bin/ocloc','cache','-show-stats','-dir','/out/offline-cache']
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (OUT/(mode+'.txt')).write_bytes(p.stdout)
 match=re.search(rb'(?m)^Version\s+Hits\s+Misses\s+Hit Rate\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+[0-9.]+%',p.stdout)
 save(mode+'.json',dict(command=cmd,exit=p.returncode,counters=dict(zip(('version','hits','misses'),map(int,match.groups()))) if match else 'UNKNOWN_UNPARSED_OR_FRESH_CACHE'))
 # Stats are observational, not a cache-hit or build qualification requirement.
else:raise AssertionError(mode)
