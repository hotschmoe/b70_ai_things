#!/usr/bin/env python3
# Lean S129 plan/stats subset; no configure or historical artifact rehash.
import json,os,re,shlex,subprocess,sys
from pathlib import Path
O=Path('/out')
def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
mode=sys.argv[1]
if mode=='before':
 names=['strata','libstrata_engine.a','libstrata_kernels.a','CMakeFiles/strata_engine.dir/src/core/verify.cpp.o','CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.dp.cpp.o','.ninja_log','.ninja_deps']
 rows={}
 for n in names:
  p=Path(n);s=p.stat();rows[n]=dict(uid=s.st_uid,gid=s.st_gid,size=s.st_size,mtime_ns=s.st_mtime_ns,writable=os.access(p,os.W_OK),parent_writable=os.access(p.parent,os.W_OK))
  assert s.st_uid==1000 and rows[n]['writable'] and rows[n]['parent_writable'],(n,rows[n])
 cache=Path('/out/offline-cache');assert cache.is_dir() and cache.stat().st_uid==1000 and os.access(cache,os.W_OK)
 save('ownership-before.json',rows)
 save('build-environment.json',dict(cwd=os.getcwd(),uid=os.getuid(),gid=os.getgid(),cache=str(cache),environment={k:os.environ[k]for k in ['TMPDIR','SYCL_CACHE_PERSISTENT','NEO_CACHE_PERSISTENT','NEO_CACHE_DIR','NEO_CACHE_STATS','MAX_JOBS','OMP_NUM_THREADS','MKL_NUM_THREADS']}))
elif mode=='plan':
 actions=[];compiled=[];archives=[];links=[]
 for line in (O/'ninja-plan.txt').read_text().splitlines():
  if not re.match(r'^\[\d+/\d+\] ',line):continue
  cmd=line.split('] ',1)[1];a=shlex.split(cmd);actions.append(cmd);assert '--regenerate-during-build'not in cmd
  if '-c'in a:
   assert a[0]=='/opt/intel/oneapi/compiler/2026.1/bin/icpx'
   assert all(x in a for x in ['-O3','-fp-model=precise','-fsycl','-fsycl-targets=spir64_gen','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel'])
   compiled.append((a[a.index('-c')+1],a[a.index('-o')+1]))
  elif '/ar 'in cmd or 'llvm-ar 'in cmd:
   names=[n for n in ['libstrata_engine.a','libstrata_kernels.a']if n in a];assert len(names)==1;archives.extend(names)
  else:
   assert '-o'in a and a[a.index('-o')+1]=='strata'
   assert '/opt/intel/oneapi/compiler/2026.1/bin/icpx'in a
   assert all(x in a for x in ['-O3','-fsycl-max-parallel-link-jobs=4','-fsycl','-fsycl-targets=spir64_gen','-fsycl-device-code-split=per_kernel'])
   assert a.count('-Xsycl-target-backend=spir64_gen')==1
   assert '-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt -allow_caching -cache_dir /out/offline-cache'in a;links.append(cmd)
 assert len(actions)==5 and sorted(compiled)==sorted([('/work/src-ep/sycl/src/core/verify.cpp','CMakeFiles/strata_engine.dir/src/core/verify.cpp.o'),('/work/src-ep/sycl/src/kernels/cuda/verify_kernels.dp.cpp','CMakeFiles/strata_kernels.dir/src/kernels/cuda/verify_kernels.dp.cpp.o')]) and sorted(archives)==['libstrata_engine.a','libstrata_kernels.a'] and len(links)==1,(actions,compiled,archives,links)
 save('root-validated-plan.json',dict(status='PASS_EXACT_FIVE',compiled=compiled,archives=archives,links=links))
elif mode in ('cache-before','cache-after'):
 cmd=['/usr/bin/ocloc','cache','-show-stats','-dir','/out/offline-cache'];p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(O/(mode+'.txt')).write_bytes(p.stdout)
 m=re.search(rb'(?m)^Version\s+Hits\s+Misses\s+Hit Rate\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+[0-9.]+%',p.stdout)
 save(mode+'.json',dict(command=cmd,exit=p.returncode,counters=dict(zip(('version','hits','misses'),map(int,m.groups())))if m else 'UNKNOWN_UNPARSED'))
else:raise AssertionError(mode)
