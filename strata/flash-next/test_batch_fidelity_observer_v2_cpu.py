#!/usr/bin/env python3
"""0024-v2 exact source reconstruction + actual header graph-record/replay mocks."""
import hashlib,json,shutil,subprocess,tempfile,os,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'batch-fidelity-observer-draft-plan-v2.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
MOCK=r'''#pragma once
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include <array>
namespace sycl {
namespace property {namespace queue {struct in_order{};}}
inline int allocations=0,frees=0,copies=0,waits=0;struct event{void wait_and_throw()const{++waits;}};
struct node{std::function<void()>fn;size_t bytes;};struct state{int context=1,device=0,key=-1;bool ordered=true;std::array<std::vector<node>,16>graphs;};
class queue {std::shared_ptr<state>s_;public:
 queue(int context=1,int device=0,bool ordered=true):s_(std::make_shared<state>()){s_->context=context;s_->device=device;s_->ordered=ordered;}
 int get_context()const{return s_->context;}int get_device()const{return s_->device;}template<class T>bool has_property()const{return s_->ordered;}void wait_and_throw()const{++waits;}
 event memcpy(void*dst,const void*src,size_t n){++copies;auto f=[=]{std::memcpy(dst,src,n);};if(s_->key<0)f();else s_->graphs[s_->key].push_back({f,n});return {};}
 void produce(std::function<void()> fn){assert(s_->key>=0);s_->graphs[s_->key].push_back({fn,0});}
 void begin(int key){s_->graphs[key].clear();s_->key=key;}void end(){s_->key=-1;}
 void replay(int key,bool omit_first_epoch=false,bool omit_stamp=false){bool skipped=false;const auto&g=s_->graphs[key];for(size_t i=0;i<g.size();++i){if(omit_stamp&&i>=g.size()-2)continue;if(omit_first_epoch&&!skipped&&g[i].bytes==8){skipped=true;continue;}g[i].fn();}}
};
template<class T>T*malloc_device(size_t n,const queue&){++allocations;return static_cast<T*>(std::calloc(n,sizeof(T)));}inline void free(void*p,const queue&){++frees;std::free(p);}
}
'''
CPP=r'''#include <cassert>
#include "strata/core/batch_fidelity_observer.hpp"
#include <iostream>
using namespace strata::core::batch_fidelity;
int main(int argc,char**argv){
 if(argc>1&&std::string(argv[1])=="off"){snapshot s(sycl::queue(),0,0,48,true);assert(!armed());assert(!begin(1,1,0,100).rid);s.release();assert(!sycl::allocations&&!sycl::copies&&!sycl::frees&&!sycl::waits);return 0;}
 unsigned rejects=0;auto bad=[&](auto f){try{f();}catch(const std::exception&){++rejects;return;}assert(false);};
 auto failed=[&](auto f){assert(!f());++rejects;};
 sycl::queue q0(1,0),q1(2,1),peer(3,1),unordered(4,0,false);bad([&]{snapshot invalid(unordered,0,0,48,true);});bad([&]{snapshot invalid(q0,0,0,32,true);});
 auto a=begin(11,1,0,100),b=begin(12,2,1,200);a.pos=99;a.token=31;b.pos=199;b.token=32;bool selected[2]={true,true};identity ids[2]={a,b};
 snapshot s0(q0,0,0,32,false),s1(q1,1,32,48,true);std::vector<float> residual(2*D),head(2*VOCAB);int colour[2]={0,1};
 int slots[2]={0,1},wrongslots[2]={1,0},migratedslots[2]={2,1};auto admission=graph_layout::make(nullptr,1,true);auto later=graph_layout::make(slots,2,false);auto wrong=graph_layout::make(wrongslots,2,false);auto migration=graph_layout::make(migratedslots,2,false);
 auto build=[&](snapshot& s,sycl::queue& q,int key,int lb,int le,bool hashead,const graph_layout& layout){int n=int(layout.words[0]);s.begin_roster(layout,q);q.begin(key);
  for(int l=lb;l<le;++l){q.produce([&,l,n]{for(int row=0;row<n;++row)std::fill(residual.begin()+row*D,residual.begin()+(row+1)*D,float(l*10+colour[row]));});s.layer(l,residual.data(),n,q);}
  if(hashead){q.produce([&,n]{for(int row=0;row<n;++row)std::fill(head.begin()+row*VOCAB,head.begin()+(row+1)*VOCAB,float(100+colour[row]));});s.logits(head.data(),n,q);}s.stamp(q);q.end();s.seal_roster(layout);
 };
 build(s0,q0,0,0,32,false,admission);build(s1,q1,0,32,48,true,admission);build(s0,q0,1,0,32,false,later);build(s1,q1,1,32,48,true,later);build(s0,q0,2,0,32,false,wrong);
 if(argc>1&&std::string(argv[1])=="coverage"){
  std::string error;for(int row=0;row<2;++row){colour[0]=row;auto id=ids[row];s0.arm(admission,&id,selected,1,true,q0);q0.replay(0);assert(s0.dump(&id,selected,1,true,error));s1.arm(admission,&id,selected,1,true,q1);q1.replay(0);assert(s1.dump(&id,selected,1,true,error));records().complete(id,true);
   std::fprintf(stderr,"SBF resume pid=%ld rid=%llu enginegen=%llu requestgen=%llu slot=%d slotgen=%llu reused=0 read_from=0 reread_to=0\n",long(getpid()),(unsigned long long)id.rid,(unsigned long long)id.enginegen,(unsigned long long)id.generation,id.slot,(unsigned long long)id.slotgen);
   for(int stage=0;stage<2;++stage){span(id,stage,stage?32:0,stage?48:32,0,id.prompt-1,"prefill_chunk");span(id,stage,stage?32:0,stage?48:32,id.prompt-1,id.prompt,"admission_target_verify");}
  }
  colour[0]=0;ids[0].pos=100;ids[1].pos=200;ids[0].event=ids[1].event=1;ids[0].row=0;ids[1].row=1;ids[0].batchrows=ids[1].batchrows=2;
  s0.arm(later,ids,selected,2,false,q0);q0.replay(1);assert(s0.dump(ids,selected,2,false,error));s1.arm(later,ids,selected,2,false,q1);q1.replay(1);assert(s1.dump(ids,selected,2,false,error));
  std::fprintf(stderr,"SBF batch_event pid=%ld enginegen=%llu event=1 rows=2 active_mask=3 completed=1\n",long(getpid()),(unsigned long long)engine_generation());
  for(int row=0;row<2;++row)for(int stage=0;stage<2;++stage)span(ids[row],stage,stage?32:0,stage?48:32,ids[row].pos,ids[row].pos+1,"batch_target_verify");
  records().complete(ids[0],false);records().complete(ids[1],false);s0.release();s1.release();std::cout<<"PASS synthetic complete2-request mock raw coverage\n";return 0;
 }
 bad([&]{s0.begin_roster(admission,q0);}); // Immutable sealed graph cannot be silently rebound.
 std::string error;
 failed([&]{return s0.dump(ids,selected,1,true,error);}); // No arm.
 s0.arm(admission,ids,selected,1,true,q0);failed([&]{return s0.dump(ids,selected,1,true,error);}); // No replay poisons frame.
 q0.replay(0);failed([&]{return s0.dump(ids,selected,1,true,error);}); // Failed epoch cannot later publish without fresh arm.
 s0.discard();s0.arm(admission,ids,selected,1,true,q0);q0.replay(0,false,true);failed([&]{return s0.dump(ids,selected,1,true,error);});
 s0.discard();s0.arm(admission,ids,selected,1,true,q0);q0.replay(0,true,false);failed([&]{return s0.dump(ids,selected,1,true,error);});
 s0.discard();s0.arm(admission,ids,selected,1,true,q0);q0.replay(0);assert(s0.dump(ids,selected,1,true,error));failed([&]{return s0.dump(ids,selected,1,true,error);});
 s1.arm(admission,ids,selected,1,true,q1);q1.replay(0);assert(s1.dump(ids,selected,1,true,error));records().complete(a,true);
 colour[0]=1;s0.arm(admission,&ids[1],selected,1,true,q0);failed([&]{return s0.dump(&ids[1],selected,1,true,error);}); // Previous RID/epoch cleared.
 s0.discard();s0.arm(admission,&ids[1],selected,1,true,q0);q0.replay(0);assert(s0.dump(&ids[1],selected,1,true,error));s1.arm(admission,&ids[1],selected,1,true,q1);q1.replay(0);assert(s1.dump(&ids[1],selected,1,true,error));records().complete(b,true);
 colour[0]=0;ids[0].pos=100;ids[1].pos=200;ids[0].event=ids[1].event=1;ids[0].row=0;ids[1].row=1;ids[0].batchrows=ids[1].batchrows=2;
 bad([&]{s0.arm(later,ids,selected,2,false,peer);});auto oldengine=ids[0];oldengine.enginegen++;identity oldengineids[2]={oldengine,ids[1]};bad([&]{s0.arm(later,oldengineids,selected,2,false,q0);});
 auto badid=ids[0];badid.pos=99;identity badids[2]={badid,ids[1]};bad([&]{s0.arm(later,badids,selected,2,false,q0);});
 s0.arm(later,ids,selected,2,false,q0);q0.replay(2);failed([&]{return s0.dump(ids,selected,2,false,error);}); // Same epoch, wrong actual row layout.
 s0.discard();s0.arm(later,ids,selected,2,false,q0);q0.replay(1);auto changed=ids[0];changed.token++;identity changedids[2]={changed,ids[1]};failed([&]{return s0.dump(changedids,selected,2,false,error);});
 s0.discard();s0.arm(later,ids,selected,2,false,q0);q0.replay(1);s1.arm(later,ids,selected,2,false,q1);q1.replay(1);s1.discard();
 auto newer=records().begin(11,3,2,100,engine_generation());newer.pos=100;newer.token=31;newer.event=1;failed([&]{return s0.dump(ids,selected,2,false,error);}); // Migration invalidates old RID/slotgen arm even if graph replays.
 assert(!records().find(11)->later);s0.discard();failed([&]{return s0.dump(ids,selected,2,false,error);});
 bad([&]{records().begin(11,1,0,100,engine_generation());}); // Stale migration cannot restore previous slot identity.
 ids[0]=newer;ids[0].batchrows=2;build(s0,q0,3,0,32,false,migration);build(s1,q1,3,32,48,true,migration);
 s0.arm(migration,ids,selected,2,false,q0);q0.replay(3);assert(s0.dump(ids,selected,2,false,error));s1.arm(migration,ids,selected,2,false,q1);q1.replay(3);assert(s1.dump(ids,selected,2,false,error));
 records().complete(newer,false);records().complete(b,false);
 // Isolated partial producer roster cannot seal or stamp and never creates an armable graph.
 {snapshot partial(q0,0,0,32,false);partial.begin_roster(admission,q0);q0.begin(4);partial.layer(0,residual.data(),1,q0);bad([&]{partial.stamp(q0);});q0.end();bad([&]{partial.seal_roster(admission);});bad([&]{partial.arm(admission,ids,selected,1,true,q0);});}
 s0.release();s1.release();assert(sycl::allocations==3&&sycl::frees==3);std::cout<<"PASS exact headers graph recording/replay, epochs/rosters/layout/context/migration, negatives="<<rejects<<"\n";
}
'''
with tempfile.TemporaryDirectory(prefix='batch-observer-v2-cpu-') as directory:
 w=Path(directory);base=Path(plan['base']);baseplan=ROOT/plan['base_plan'];assert sha(baseplan)==plan['base_plan_sha256']
 for rel,h in json.loads(baseplan.read_text())['expected_patched_source_sha256'].items():assert sha(base/rel)==h
 shutil.copytree(base,w/'source');source=w/'source'
 for item in plan['patches']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];subprocess.run(['git','apply','--check',str(p)],cwd=source,check=True);subprocess.run(['git','apply',str(p)],cwd=source,check=True)
 for rel,h in plan['source_sha256'].items():assert sha(source/rel)==h
 v=(source/'sycl/src/core/verify.cpp').read_text();g=(source/'sycl/src/program/generate.cpp').read_text()
 assert v.count('batch_admission_exec_[ar_off_?1:0]')==2 and v.count('(batch_observe_ ? batch_observe_exec_ : exec_bm_)')==2
 assert v.index('batch_snapshot_->logits(head_logits_,T,*cs)')<v.index('argmax_rows(head_logits_, T')<v.index('batch_snapshot_->stamp(*cs)')
 assert 'actual rows/positions/tokens disagree' in v and 'actual admission token/position mismatch' in v and 'uint64_t(bt_windows+1)' in g
 # Begin descriptor upload precedes actual queue capture, and arm precedes the real graph launch.
 batch=v[v.index('bool Verifier::capture_batch('):v.index('bool Verifier::capture_commit_batch(')]
 assert batch.index('batch_snapshot_->begin_roster')<batch.index('begin_recording') and batch.index('end_recording')<batch.index('seal_roster')
 mock=w/'mock/sycl';mock.mkdir(parents=True);(mock/'sycl.hpp').write_text(MOCK);(w/'test.cpp').write_text(CPP);capture=w/'capture';capture.mkdir();coverdir=w/'coverage';coverdir.mkdir();(w/'ARM').write_text('ARM\n')
 command='g++ -std=c++17 -Wall -Wextra -Werror -Wno-misleading-indentation -Wno-parentheses -fsanitize=address,undefined -g -I/work/mock -I/work/source/sycl/include /work/test.cpp -o /work/test && env -u STRATA_BATCH_FIDELITY_DIAG -u STRATA_BATCH_FIDELITY_ARM -u STRATA_BATCH_FIDELITY_DIR /work/test off && STRATA_BATCH_FIDELITY_DIAG=1 STRATA_BATCH_FIDELITY_ARM=/work/ABSENT STRATA_BATCH_FIDELITY_DIR=/work/capture /work/test off && STRATA_BATCH_FIDELITY_DIAG=1 STRATA_BATCH_FIDELITY_ARM=/work/ARM STRATA_BATCH_FIDELITY_DIR=/work/capture /work/test && STRATA_BATCH_FIDELITY_DIAG=1 STRATA_BATCH_FIDELITY_ARM=/work/ARM STRATA_BATCH_FIDELITY_DIR=/work/coverage /work/test coverage'
 result=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc',command],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 files=list(capture.glob('*.f32'));assert len(files)==196,'Failed replays must not create any raw files'
 for p in files:
  raw=p.read_bytes();layer=int(p.name.split('-l')[1].split('-i')[0]);colour=0 if '-r11-' in p.name else 1;expected=100+colour if layer==-1 else layer*10+colour
  assert len(raw)==(248320 if layer==-1 else 10240)*4 and all(x[0]==expected for x in struct.iter_unpack('<f',raw))
 import importlib.util
 spec=importlib.util.spec_from_file_location('coveragev2',HERE/'audit_batch_fidelity_coverage_v2.py');auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
 # Separate successful process has its own engine generation and files. Earlier negative controls remain separate.
 trace=result.stderr[result.stderr.rindex('SBF request pid='):]
 # rindex selects second request; find first request of last engine generation instead.
 requests=[line for line in result.stderr.splitlines() if line.startswith('SBF request')];last_engine=dict(x.split('=',1) for x in requests[-1].split()[2:] if '=' in x)['enginegen'];engine_start=next(line for line in requests if 'enginegen='+last_engine+' ' in line);trace=result.stderr[result.stderr.index(engine_start):]
 report=auditor.coverage(trace,[11,12],[(0,0,32),(1,32,48)],coverdir);assert report['vectors']==196 and report['replay_proofs']==6 and report['completed_batch_events']==1
 lines=trace.splitlines();vec=next(line for line in lines if line.startswith('SBF vector'));proof=next(line for line in lines if line.startswith('SBF replay'));event=next(line for line in lines if line.startswith('SBF batch_event'))
 def reject(text):
  try:auditor.coverage(text,[11,12],[(0,0,32),(1,32,48)],coverdir)
  except (AssertionError,ValueError,KeyError):return
  raise AssertionError('Unqualified raw replay coverage accepted')
 reject('');reject(trace.replace(vec,'',1));reject(trace+'\n'+vec);reject(trace.replace(proof,'',1));reject(trace+'\n'+proof);reject(trace.replace(event,'',1));reject(trace.replace('rows=2 active_mask=3','rows=2 active_mask=5',1));reject(trace.replace('full_roster=1','full_roster=0',1));reject(trace.replace('input_output_verified=1','input_output_verified=0',1));reject(trace.replace('stage_context_verified=1','stage_context_verified=0',1));reject(trace.replace('enginegen='+last_engine,'enginegen=0',1));reject(trace.replace('slotgen=1','slotgen=9',1));reject(trace.replace('begin=0 end=99','begin=1 end=99',1));reject(trace+'\nSBF bound span_events=8192 exceeded=1');rowproof=next(line for line in lines if line.startswith('SBF row'));reject(trace.replace(rowproof,'',1));reject(trace+'\n'+rowproof);epoch=dict(word.split('=',1) for word in vec.split()[2:] if '=' in word)['epoch'];reject(trace.replace(vec,vec.replace('epoch='+epoch,'epoch='+str(int(epoch)+1)),1))
 print(result.stdout.strip());print('PASS strict synthetic collector complete196 vectors/6 stage replays/1 actual-event fixture +17 empty/missing/duplicate/epoch/row/roster/context/engine/slot/mask/span/bound negatives')
 print('PASS CPU tracked final21+22v2+24-v2 source and defaultoff/unarmed zeroalloc/copy; no full SYCL TU/SDK/GPU/model/concurrency qualification.')
