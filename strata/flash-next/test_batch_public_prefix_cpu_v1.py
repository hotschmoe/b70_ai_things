#!/usr/bin/env python3
"""Public27 actual source policy/reset/native guards and API wire fields, CPU-only."""
import ast,hashlib,json,os,shutil,subprocess,tempfile,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'batch-public-prefix-source-plan-v1.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(text,start):
 i=text.index(start);op=text.index('{',i);depth=0;quote=None;escaped=False;comment=None;j=op
 while j<len(text):
  c=text[j];pair=text[j:j+2]
  if comment=='line':
   if c=='\n':comment=None
  elif comment=='block':
   if pair=='*/':comment=None;j+=1
  elif quote:
   if escaped:escaped=False
   elif c=='\\':escaped=True
   elif c==quote:quote=None
  elif pair=='//':comment='line';j+=1
  elif pair=='/*':comment='block';j+=1
  elif c in ('"',"'"):quote=c
  elif c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth==0:return text[i:j+1]+(';' if start.startswith('auto ') else '')
  j+=1
 raise AssertionError('Source body incomplete')
CPP=r'''#include <cassert>
#include "strata/core/batch_public_prefix.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/core/batch_prefix_chain.hpp"
#include "strata/program/conv_cache.hpp"
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <iostream>
namespace sycl {struct exception:std::runtime_error{using std::runtime_error::runtime_error;};}
static int device=0,writes=0,fail_device=-1;
struct Queue {int dev=0,context=0;Queue(int d=0):dev(d),context(d+1){}int get_device()const{return dev;}int get_context()const{return context;}struct Event{void wait_and_throw(){}};
 Event memset(void*p,int value,size_t n){if(dev==fail_device)throw sycl::exception("mock queue reset failure");if(n){++writes;std::memset(p,value,n);}return {};}
 Event memcpy(void*p,const void*s,size_t n){if(dev==fail_device)throw sycl::exception("mock queue copy failure");if(n){++writes;std::memcpy(p,s,n);}return {};}
 void wait_and_throw(){}void wait(){}
};
namespace dpct {using queue_ptr=Queue*;inline Queue&get_in_order_queue(){static Queue queues[2]={{0},{1}};return queues[device];}}
namespace strata::kernels {constexpr int kStepCount=4,KV_Q8_GROUP=64;inline size_t kv_q4_bytes_per_head(int n){return size_t(n);};inline void kv_stream_reset(int,void*){}}
namespace strata {inline dpct::queue_ptr q_of(void*s){return static_cast<Queue*>(s);}inline void big_fill_zero(Queue&q,void*p,size_t n){q.memset(p,0,n);}}
namespace strata::core {
struct OnDevice {int old;OnDevice(int d):old(device){device=d;}~OnDevice(){device=old;}};
struct ModelGeometry {int n_head=1,hc=4,n_embd=1;};struct QsaShapes {int n_head_kv=1,page_size=4,head_dim=4,idx_block=4,idx_dim=1;};inline QsaShapes qsa_shapes(const ModelGeometry&){return {};}
struct QsaState {int kv_mode=0,kv_elastic=-1,map=0;bool kv_hybrid=false,kv_q4=false,kv_int8=false;int64_t n_slots=4,n_pages=4,idx_pooled_rows=6;uint16_t*k_pool=nullptr,*v_pool=nullptr,*k_scale=nullptr,*v_scale=nullptr;uint8_t*k_q=nullptr,*v_q=nullptr,*k_q4=nullptr,*v_q4=nullptr;float*idx_tail=nullptr,*idx_dead=nullptr,*idx_pooled=nullptr;int32_t*idx_block_pos=nullptr,*step=nullptr,*attention_status=nullptr,*pos_dev=nullptr,*page_table=nullptr,*host_step=nullptr,*host_pos=nullptr;};
struct SessionState {struct Block{float*R=nullptr;}block;float*gdn_state=nullptr,*ple_hist=nullptr;int64_t gdn_alloc=1,qsa_alloc=1,qsa_ord0=1;QsaState*qsa_states=nullptr;int32_t ple_prev[2]={99,99},ple_token=99;};
inline size_t gdn_state_floats(const ModelGeometry&){return 2;}inline size_t ple_hist_bytes(){return 4;}struct Pool{};inline std::vector<Pool*>g_pools;inline int pool_slots_mapped(const Pool&){return 4;}
'''
AFTER=r'''
}
struct Owned {strata::core::SessionState ss;std::array<float,4>R;std::array<float,2>gdn;float ple,dead;std::array<float,3>tail;std::array<float,6>pooled;int32_t block,status;std::array<int32_t,4>step,pages;std::array<int32_t,5>hstep;std::array<int32_t,1>pos,hpos;std::array<uint16_t,64>k,v;std::array<strata::core::QsaState,2>q;
 Owned(){ss.block.R=R.data();ss.gdn_state=gdn.data();ss.ple_hist=&ple;ss.qsa_states=q.data();auto&x=q[1];x.k_pool=k.data();x.v_pool=v.data();x.idx_tail=tail.data();x.idx_dead=&dead;x.idx_pooled=pooled.data();x.idx_block_pos=&block;x.step=step.data();x.attention_status=&status;x.pos_dev=pos.data();x.page_table=pages.data();x.host_step=hstep.data();x.host_pos=hpos.data();fill();}
 void fill(){R.fill(42);gdn.fill(42);ple=dead=42;tail.fill(42);pooled.fill(42);block=status=42;step.fill(42);pages.fill(42);hstep.fill(42);pos.fill(42);hpos.fill(42);k.fill(42);v.fill(42);ss.ple_prev[0]=ss.ple_prev[1]=ss.ple_token=99;}
 bool zero(const strata::core::SessionState& state)const{return gdn==std::array<float,2>{}&&R==std::array<float,4>{}&&ple==0&&dead==0&&tail==std::array<float,3>{}&&pooled==std::array<float,6>{}&&block==0&&status==0&&step==std::array<int32_t,4>{}&&hstep==std::array<int32_t,5>{}&&pos[0]==0&&hpos[0]==0&&k==std::array<uint16_t,64>{}&&v==std::array<uint16_t,64>{}&&pages==std::array<int32_t,4>{0,1,2,3}&&state.ple_prev[0]==-1&&state.ple_prev[1]==-1&&state.ple_token==-1;}
};
struct Stage {int dev=1;strata::core::SessionState ss;Queue*stream=nullptr;};using ConvCheckpoint=strata::core::ConversationCheckpoint;
inline bool checkpoint_save(ConvCheckpoint&c,const strata::core::SessionState&s,const strata::core::ModelGeometry&){const auto*p=reinterpret_cast<const uint8_t*>(s.gdn_state);c.gdn.assign(p,p+8);return true;}
struct Verify{bool wait_commit(std::string&){return true;}};
int main(){using namespace strata::core;using namespace strata::core::batch_public;
 assert(enabled());int rejected=0;auto bad=[&](auto fn){try{fn();}catch(const std::exception&){++rejected;return;}assert(false);};
 assert(ceiling(true,false,0,13)==0&&ceiling(false,true,0,13)==0&&ceiling(false,true,5,13)==5&&ceiling(false,true,12,13)==12&&ceiling(false,false,0,13)==12);bad([&]{ceiling(false,true,13,13);});bad([&]{ceiling(false,true,-1,13);});
 ConversationCache cache(1<<20,2);SavedConversation image;for(int i=1;i<=11;++i)image.live.ids.push_back(i);ConversationCheckpoint root;for(int i=1;i<=3;++i)root.ids.push_back(i);image.checkpoints.push_back(root);auto pin=root;pin.ids.push_back(4);pin.ids.push_back(5);image.checkpoints.push_back(pin);assert(cache.put(std::move(image)));std::vector<int64_t>prompt;for(int i=1;i<=13;++i)prompt.push_back(i);
 assert(cache.best(prompt,{},true).tokens==11);assert(cache.best(prompt,{},true,5).tokens==5);assert(cache.best(prompt,{},true,4).tokens==3);assert(cache.best(prompt,{},true,0).tokens==0);
 Owned a,b,active;auto&ss=a.ss;Queue q0(0),q1(1),badpeer(1);badpeer.context=999;Queue*main_stream=&q0;std::vector<std::unique_ptr<Stage>>stages;stages.push_back(std::make_unique<Stage>());stages[0]->ss=b.ss;stages[0]->stream=&q1;ModelGeometry g;Verify ver;std::vector<int32_t>live={1,2,3};std::vector<ConversationImageKey>live_imgs;std::vector<ConversationCheckpoint>checks={root};bool live_ok=true;std::string error;std::string&err=error;
 bool batch_chain_enabled=true;size_t batch_chain_point_bound=1<<20;auto batch_chain_room=[](size_t){return true;};std::string ckpt_why;uint64_t check_clock=0;int64_t tail_ckpt_len=-1;struct Options{int prompt_cache=3;}o;std::vector<int64_t>cur;for(int i=1;i<=13;++i)cur.push_back(i);std::vector<strata::core::ConversationImageKey>req_imgs;auto imgs_below=[](const auto&,int64_t){return std::vector<strata::core::ConversationImageKey>{};};auto gpu_sync_ok=[](){return true;};
'''
TEST=r'''
 int before=writes;stages[0]->stream=&badpeer;assert(!public_reset_main(error));assert(writes==before);++rejected;stages[0]->stream=&q1;
 auto*old=b.q[1].idx_block_pos;b.q[1].idx_block_pos=nullptr;before=writes;assert(!public_reset_main(error));assert(writes==before);++rejected;b.q[1].idx_block_pos=old;
 assert(public_reset_main(error));assert(a.zero(ss)&&b.zero(stages[0]->ss)&&live.empty()&&checks.empty()&&!live_ok);assert(active.gdn[0]==42&&active.block==42&&active.pages[0]==42&&active.ss.ple_prev[0]==99);
 // Exact old session_zero leaves block/staging metadata; public wrapper clears it on every stage.
 a.fill();strata::core::session_zero(ss,g,nullptr,&q0);q0.wait();assert(a.gdn[0]==0&&a.dead==0&&a.block==42&&a.step[0]==42&&a.pages[1]==42);
 a.fill();b.fill();live={1,2,3};checks={root};live_ok=true;fail_device=1;assert(!public_reset_main(error));++rejected;assert(!live.empty()&&!checks.empty()&&live_ok&&active.gdn[0]==42);fail_device=-1;assert(public_reset_main(error));assert(a.zero(ss)&&b.zero(stages[0]->ss));
 // Pin below deepest11 selects actual root3, evaluates to5, then captures real state before later rows.
 checks.clear();a.gdn[0]=6;b.gdn[0]=12;int64_t actual_consumed=3;std::vector<int>evaluated;
 for(int token=4;token<=5;++token){a.gdn[0]+=float(token);b.gdn[0]+=float(token*2);actual_consumed=token;evaluated.push_back(token);}
 assert(actual_consumed==5&&checkpoint_at(5,nullptr,false,true));assert(checks.size()==1&&checks[0].pinned&&checks[0].ids==std::vector<int32_t>({1,2,3,4,5})&&checks[0].stage_parts.size()==1&&checks[0].stage_parts[0].ids==checks[0].ids);
 float captured=0,peer=0;std::memcpy(&captured,checks[0].gdn.data(),4);std::memcpy(&peer,checks[0].stage_parts[0].gdn.data(),4);assert(captured==15&&peer==30&&evaluated==std::vector<int>({4,5}));
 a.gdn[0]=78;b.gdn[0]=156;assert(checkpoint_at(12,nullptr,false,true));assert(checks.back().ids.size()==12&&checks.back().pinned);
 std::cout<<"PASS actual session_zero/QSAzero/publicreset source: allmainstages persistentstate+metadata cleared, activepeer unchanged; no metadata-only fresh; controls="<<rejected<<"\n";
}
'''
with tempfile.TemporaryDirectory(prefix='public-prefix27-cpu-') as name:
 w=Path(name);base=Path(plan['compiled20_base']);baseline=ROOT/plan['compiled20_plan'];assert sha(baseline)==plan['compiled20_plan_sha256']
 for rel,h in json.loads(baseline.read_text())['expected_patched_source_sha256'].items():assert sha(base/rel)==h
 shutil.copytree(base,w/'source');source=w/'source'
 for item in plan['patches']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];subprocess.run(['git','apply','--check',str(p)],cwd=source,check=True);subprocess.run(['git','apply',str(p)],cwd=source,check=True)
 for rel,h in plan['source_sha256'].items():assert sha(source/rel)==h
 generate=(source/'sycl/src/program/generate.cpp').read_text();session=(source/'sycl/src/core/session.cpp').read_text();layer=(source/'sycl/src/core/layer.cpp').read_text();api=(source/'serve/server.py').read_text()
 # Authoritative ceilings guard EVERY selection route. Pin is captured at cut before later prompt rows.
 assert 'if (cache_prefix_ceiling>0 && o.prompt_cache > 0' in generate and 'cache_prefix_ceiling>0?conversations.best' in generate and '!req_fresh && (!from_live || incoming' in generate
 assert 'L > cache_prefix_ceiling' in generate and 'to == pin_at' in generate and 'before_later_rows=1' in generate
 cpp=CPP+extract(layer,'void qsa_state_zero(')+'\n'+extract(session,'void session_zero(')+AFTER+extract(generate,'auto public_reset_main=')+extract(generate,'auto checkpoint_at =')+TEST;(w/'test.cpp').write_text(cpp)
 command='g++ -std=c++20 -Wall -Wextra -Werror -Wno-misleading-indentation -fsanitize=address,undefined -fno-sanitize-recover=all -g -I/work/source/sycl/include -I/work/source/include /work/test.cpp -o /work/test && STRATA_BATCH_PUBLIC_PREFIX=1 /work/test'
 result=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc',command],capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
 tree=ast.parse(api);engine=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StrataEngine');methods=[n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name in ('sampling_keys','projection_key')];unit=ast.Module(body=[ast.ClassDef(name='StrataEngine',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(unit);ns={};exec(compile(unit,'actual27-api','exec'),ns);keys=ns['StrataEngine'].sampling_keys
 assert keys({'strata_fresh':True,'strata_prefix':{'tokens':0}})==' fresh=1 pin=0';assert 'pin=12' in keys({'strata_shared_prefix':{'tokens':12}})
 for bad in [{'strata_fresh':1},{'strata_prefix':{'tokens':True}},{'strata_prefix':{'tokens':-1}},{'strata_prefix':{'tokens':1},'strata_shared_prefix':{'tokens':2}}]:
  try:keys(bad)
  except ValueError:pass
  else:raise AssertionError('Invalid public fields accepted')
 service=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Service');resolve=next(n for n in service.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_prefix');unit=ast.Module(body=[ast.ClassDef(name='Service',bases=[],keywords=[],body=[resolve],decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(unit);ns={'mark_think_literals':lambda *a:None};exec(compile(unit,'actual27-token-boundary','exec'),ns);obj=ns['Service']();ids=list(range(13));assert obj.resolve_prefix({'tokens':0},[],None,{},ids)=={'tokens':0};assert obj.resolve_prefix({'tokens':12},[],None,{},ids)=={'tokens':12};assert ids==list(range(13))
 guard_start=generate.index('            const bool public_batch_request=');guard_end=generate.index('            if(strict_batch_identity &&',guard_start);native_guard=generate[guard_start:guard_end]
 guard_cpp='''#include <cstdio>
#include <cstdint>
#include <cassert>
struct Options{int batch=2,prompt_cache=3,pipeline_windows=0;}o;
bool accepted(bool batch_public_enabled,bool strict_batch_identity,int64_t req_pin,bool req_pin_present,bool req_fresh,int64_t n,bool req_pin_bad=false,bool req_fresh_bad=false){bool req_ckpt=true,use_mtp=false,geni=false;for(int iteration=0;iteration<1;++iteration){'''+native_guard+'''return true;}return false;}
int main(){assert(accepted(true,true,0,true,false,13));assert(accepted(true,true,12,true,false,13));assert(accepted(true,true,0,false,true,13));assert(!accepted(false,true,5,true,false,13));assert(!accepted(false,true,0,false,true,13));assert(!accepted(true,true,13,true,false,13));o.prompt_cache=2;assert(!accepted(true,true,5,true,false,13));assert(accepted(true,true,0,true,true,13));}
'''
 (w/'guard.cpp').write_text(guard_cpp)
 guardrun=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc','g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined /work/guard.cpp -o /work/guard && /work/guard'],capture_output=True,text=True);assert guardrun.returncode==0,guardrun.stdout+guardrun.stderr
 for n in (-1,13,True):
  try:obj.resolve_prefix({'tokens':n},[],None,{},ids)
  except ValueError:pass
  else:raise AssertionError('Out-of-range/bool token pin accepted')
 print(result.stdout.strip());print('PASS actual API fresh/pin0/prompt-1 fields+prefixceilings and malformed/prefix controls. No SDK/GPU/model payload read or cache qualification.')
