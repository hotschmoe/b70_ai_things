#!/usr/bin/env python3
"""CPU-only ledger/retention controls. No GPU execution or cache math claim."""
import json,os,subprocess,tempfile,shlex,shutil
from pathlib import Path
BASE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T065210Z-e6m4q9uo/source')
HEADER=Path('/mnt/vm_8tb/b70/build/strata-prefix-diag-source-20261009/sycl/include')
with tempfile.TemporaryDirectory(prefix='strata-prefix-ledger-') as td:
 d=Path(td);src=d/'test.cpp';exe=d/'test'
 src.write_text(r'''#include "strata/program/prefix_diagnostic.hpp"
#include "strata/program/conv_cache.hpp"
#include <cassert>
int main(){
 strata::program::PrefixDiagnostic x;x.begin(std::vector<int64_t>(11,42),2);
 x.selection(8,0,false,-1,100,1,0,1);x.evaluated(0,8,"prefill");x.evaluated(8,10,"prefill");x.evaluated(10,11,"verify");x.evaluated(11,12,"verify");x.finish(false,"stop",2,100,1,0);
 uint64_t stamps[]={1,2,3,4};bool tail[]={false,false,true,false};bool pin[]={false,true,false,false};
 assert(strata::program::conv_cache::eviction_victim(stamps,4,3,tail,pin)==2);
 bool none[]={false,false,false,false};
 assert(strata::program::conv_cache::eviction_victim(stamps,4,3,none,pin)==2);
 for(int i=0;i<100;i++){strata::program::PrefixDiagnostic a;a.begin(std::vector<int64_t>(11,42),2);a.finish(false,"stop",0,0,0,0);}
}
''',encoding='ascii')
 compile_command=['g++','-std=c++20','-I'+str(HEADER),'-I'+str(BASE/'include'),str(src),'-o',str(exe)]
 if not shutil.which('g++'):
  compile_command[0]='icpx'
  compile_command=['docker','run','--rm','--network','none','--entrypoint','/bin/bash','-v',str(d)+':'+str(d),'-v',str(HEADER)+':'+str(HEADER)+':ro','-v',str(BASE/'include')+':'+str(BASE/'include')+':ro','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc',shlex.join(compile_command)]
 subprocess.run(compile_command,check=True)
 arm=d/'arm';arm.touch();env=os.environ.copy();env.update(STRATA_PREFIX_DIAG='1',STRATA_PREFIX_DIAG_ARM=str(arm))
 r=subprocess.run([str(exe)],env=env,capture_output=True,text=True,check=True)
 rows=[json.loads(s[len('PREFIX_DIAG '):]) for s in r.stderr.splitlines()]
 f=next(x for x in rows if x['event']=='finish');assert f['actual_reused']==0 and f['candidate_resume']==8
 assert (f['evaluated_prompt_rows'],f['evaluated_decode_rows'],f['stage_prompt_rows'])==(11,1,22)
 assert len([x for x in rows if x['event']=='finish'])==64
 for value,present in [('0',True),('1',False)]:
  e=env.copy();e['STRATA_PREFIX_DIAG']=value
  if not present:e['STRATA_PREFIX_DIAG_ARM']=str(d/'absent')
  r=subprocess.run([str(exe)],env=e,capture_output=True,text=True,check=True);assert not r.stderr
 print('PASS: completed prompt/decode/stage span totals, reread candidate distinction, 64-request arming bound, default off/missing ARM, pinned retention controls')
