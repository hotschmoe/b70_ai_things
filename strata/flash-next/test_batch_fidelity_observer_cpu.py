#!/usr/bin/env python3
"""Tracked0024 reconstruction and actual headers with host-only queue mocks."""
import hashlib,json,shutil,subprocess,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'batch-fidelity-observer-draft-plan.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
MOCK=r'''#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace sycl {inline unsigned allocations=0,copies=0,frees=0;struct event{void wait_and_throw(){}};
struct queue {int ctx=1,dev=0;queue(int d=0):dev(d){}int get_context()const{return ctx;}int get_device()const{return dev;}void wait_and_throw(){}event memcpy(void* a,const void* b,size_t n){++copies;std::memcpy(a,b,n);return {};}};
template<class T>T* malloc_device(size_t n,const queue&){++allocations;return static_cast<T*>(std::malloc(n*sizeof(T)));}inline void free(void* p,const queue&){++frees;std::free(p);}}
'''
CPP=r'''#include "strata/core/batch_fidelity_observer.hpp"
#include <cassert>
#include <fstream>
using namespace strata::core::batch_fidelity;
int main(int argc,char**argv){
 if(argc>1 && std::string(argv[1])=="off"){assert(!armed());assert(!begin(1,1,0,100).rid);assert(!sycl::allocations&&!sycl::copies);return 0;}
 assert(armed());ledger l;unsigned rejects=0;auto bad=[&](auto f){try{f();}catch(const std::exception&){++rejects;return;}assert(false);};
 bad([&]{l.begin(0,1,0,100);});auto id=l.begin(1,1,0,100);assert(l.select(id,true)&&!l.select(id,false));
 auto stale=id;stale.slotgen=2;bad([&]{l.select(stale,true);});stale=id;stale.generation=7;bad([&]{l.select(stale,true);});
 l.complete(id,true);assert(!l.select(id,true)&&l.select(id,false));l.complete(id,false);assert(!l.select(id,false));bad([&]{l.complete(id,false);});
 for(int i=2;i<=6;++i)assert(l.begin(i,i,i-1,100).rid);assert(!l.begin(7,1,0,100).rid);
 auto a=begin(11,1,0,100),b=begin(12,2,1,200);a.pos=99;a.token=31;b.pos=199;b.token=32;a.batchrows=b.batchrows=2;b.row=1;
 identity ids[2]={a,b};bool selected[2]={true,true};std::string error;sycl::queue q0(0),q1(1),foreign(2);
 {snapshot s0(q0,0,0,32,false),s1(q1,1,32,48,true);std::vector<float> residual(2*D),head(2*VOCAB);
 for(int layer=0;layer<48;++layer){for(int row=0;row<2;++row)std::fill(residual.begin()+row*D,residual.begin()+(row+1)*D,float(layer*10+row));if(layer<32)s0.layer(layer,residual.data(),2,q0);else s1.layer(layer,residual.data(),2,q1);}
 for(int row=0;row<2;++row)std::fill(head.begin()+row*VOCAB,head.begin()+(row+1)*VOCAB,float(100+row));s1.logits(head.data(),2,q1);
 bad([&]{s0.layer(0,residual.data(),2,foreign);});bad([&]{s0.layer(32,residual.data(),2,q0);});bad([&]{s1.logits(head.data(),7,q1);});
 assert(s0.dump(ids,selected,2,true,error));assert(s1.dump(ids,selected,2,true,error));
 ids[0].pos=100;ids[1].pos=200;assert(s0.dump(ids,selected,2,false,error));assert(s1.dump(ids,selected,2,false,error));s0.release();s1.release();}
 assert(sycl::allocations==2&&sycl::frees==2&&rejects==7);return 0;
}
'''
with tempfile.TemporaryDirectory(prefix='batch-observer-cpu-') as directory:
 w=Path(directory);base=Path(plan['base']);baseplan=ROOT/plan['base_plan'];assert sha(baseplan)==plan['base_plan_sha256']
 for rel,h in json.loads(baseplan.read_text())['expected_patched_source_sha256'].items():assert sha(base/rel)==h
 shutil.copytree(base,w/'source');source=w/'source'
 for item in plan['patches']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];subprocess.run(['git','apply','--check',str(p)],cwd=source,check=True);subprocess.run(['git','apply',str(p)],cwd=source,check=True)
 for rel,h in plan['source_sha256'].items():assert sha(source/rel)==h
 v=(source/'sycl/src/core/verify.cpp').read_text();g=(source/'sycl/src/program/generate.cpp').read_text()
 assert v.count('batch_admission_exec_[ar_off_?1:0]')==2
 assert v.count('(batch_observe_ ? batch_observe_exec_ : exec_bm_)')==2
 assert v.index('batch_snapshot_->logits(head_logits_,T,*cs)')<v.index('argmax_rows(head_logits_, T')
 assert 'if(batch_observe_ && batch_snapshot_ && grp==0)' in v
 assert 'enabled && S>=2' in g and 'select_batch_observation(batch_ids.data(),S,false' in g
 assert v.index('release_batch_observer();')<v.index('bm_used_.clear();slots_.clear();')
 mock=w/'mock/sycl';mock.mkdir(parents=True);(mock/'sycl.hpp').write_text(MOCK);(w/'test.cpp').write_text(CPP)
 capture=w/'capture';capture.mkdir();arm=w/'ARM';arm.write_text('ARM\n')
 command='g++ -std=c++17 -Wall -Wextra -Werror -Wno-misleading-indentation -fsanitize=address,undefined -g -I/work/mock -I/work/source/sycl/include /work/test.cpp -o /work/test && env -u STRATA_BATCH_FIDELITY_DIAG -u STRATA_BATCH_FIDELITY_ARM -u STRATA_BATCH_FIDELITY_DIR /work/test off && STRATA_BATCH_FIDELITY_DIAG=1 STRATA_BATCH_FIDELITY_ARM=/work/ABSENT STRATA_BATCH_FIDELITY_DIR=/work/capture /work/test off && STRATA_BATCH_FIDELITY_DIAG=1 STRATA_BATCH_FIDELITY_ARM=/work/ARM STRATA_BATCH_FIDELITY_DIR=/work/capture /work/test'
 result=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc',command],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 # Feed exact helper-produced raw files to a strict synthetic source-span contract.
 import importlib.util
 spec=importlib.util.spec_from_file_location('batchcoverage',HERE/'audit_batch_fidelity_coverage.py');auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
 trace=result.stderr
 lines=[]
 for line in trace.splitlines():
  if line.startswith('SBF vector') and 'phase=admission_' in line:
   line=line.replace('row=1 batchrows=2','row=0 batchrows=1').replace('row=0 batchrows=2','row=0 batchrows=1')
  lines.append(line)
 for rid,gen,sg,slot,prompt in [(11,1,1,0,100),(12,2,2,1,200)]:
  prefix=f'SBF resume pid=1 rid={rid} requestgen={gen} slot={slot} slotgen={sg}'
  lines.append(prefix+' reused=0 read_from=0 reread_to=0')
  for stage,lb,le in [(0,0,32),(1,32,48)]:
   prefix=f'SBF span pid=1 rid={rid} requestgen={gen} slot={slot} slotgen={sg} stage={stage} lb={lb} le={le}'
   for phase,a,b in [('prefill_chunk',0,prompt-1),('admission_target_verify',prompt-1,prompt),('batch_target_verify',prompt,prompt+1)]:lines.append(prefix+f' begin={a} end={b} phase={phase} completed=1')
 # Container PID is normally1. Derive it from actual request labels rather than assume namespace options.
 pid=next(line.split('pid=')[1].split()[0] for line in lines if line.startswith('SBF request'))
 lines=[line.replace('pid=1 ',f'pid={pid} ') for line in lines];trace='\n'.join(lines)
 assert auditor.coverage(trace,[11,12],[(0,0,32),(1,32,48)],capture)['vectors']==196
 def reject(text):
  try:auditor.coverage(text,[11,12],[(0,0,32),(1,32,48)],capture)
  except (AssertionError,ValueError,KeyError):return
  raise AssertionError('Incomplete/stale batch raw coverage accepted')
 reject('');vectors=[line for line in lines if line.startswith('SBF vector')]
 reject(trace.replace(vectors[0],'',1));reject(trace+'\n'+vectors[0]);reject(trace.replace('slotgen=1','slotgen=9',1));wrong_stage=next(line for line in vectors if 'stage=1 lb=32 le=48' in line);reject(trace.replace(wrong_stage,wrong_stage.replace('stage=1 lb=32 le=48','stage=0 lb=32 le=48'),1));reject(trace.replace('phase=batch_step_logits_before_sampler','phase=admission_logits_before_sampler',1));reject(trace.replace('batchrows=2','batchrows=1'));reject(trace.replace('begin=0 end=99','begin=1 end=99',1));reject(trace+'\nSBF bound span_events=8192 exceeded=1')

 files=list(capture.glob('*.f32'));assert len(files)==196
 import struct
 for p in files:
  raw=p.read_bytes();layer=int(p.name.split('-l')[1].split('-i')[0]);row=0 if '-r11-' in p.name else 1
  expected=100+row if layer==-1 else layer*10+row
  assert len(raw)==(248320 if layer==-1 else 10240)*4 and all(value[0]==expected for value in struct.iter_unpack('<f',raw))
 print('PASS tracked final21+22v2+24 reconstruction; exact consumed helper mock ASan/UBSan, off zero allocation/copy, full48/head row separation and bounds/stale identity/affinity negatives. No full SYCL TU/compiler/GPU or concurrency qualification.')
