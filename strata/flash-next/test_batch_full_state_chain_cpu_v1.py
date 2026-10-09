#!/usr/bin/env python3
"""Source-bound0025 policy and actual transfer/restore bodies, ordinary host C++ mocks."""
import hashlib,json,shutil,subprocess,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'batch-full-state-chain-source-plan-v1.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(text,start):
 i=text.index(start);opening=text.index('{',i);depth=0;quote=None;escaped=False;comment=None;j=opening
 while j<len(text):
  c=text[j];following=text[j:j+2]
  if comment=='line':
   if c=='\n':comment=None
  elif comment=='block':
   if following=='*/':comment=None;j+=1
  elif quote:
   if escaped:escaped=False
   elif c=='\\':escaped=True
   elif c==quote:quote=None
  elif following=='//':comment='line';j+=1
  elif following=='/*':comment='block';j+=1
  elif c in ('"',"'"):quote=c
  elif c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth==0:return text[i:j+1]+(';' if start.startswith('auto ') else '')
  j+=1
 raise AssertionError('Unterminated source block')
PRE=r'''#include <cassert>
#include "strata/core/batch_prefix_chain.hpp"
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <cstring>
using ConvCheckpoint=strata::core::ConversationCheckpoint;
namespace sycl {struct exception:std::exception {const char*what()const noexcept override{return "mock SYCL failure";}};}
static int device=0,writes=0,saves=0,waits=0,restore_fail=-1,change_at_wait=-1;static bool memory_ok=true;static int parked_captures=0;static std::function<void()>changed;
namespace dpct {struct Device {int queues_wait_and_throw(){++waits;if(waits==change_at_wait&&changed)changed();return 0;}};inline Device&get_current_device(){static Device d;return d;}}
#define DPCT_CHECK_ERROR(expr) (expr)
namespace strata::kernels {struct Shapes{int idx_block=4;};inline Shapes qsa_real_shapes(){return {};}}
namespace strata::core {
struct OnDevice {int old;OnDevice(int d):old(device){device=d;}~OnDevice(){device=old;}};
struct ModelGeometry {int idx_key_dim=1;};struct ConversationStateSizes {size_t gdn=8,ple=4,tail=4,dead=4,block_pos=4;};
struct QsaState {int64_t max_cells=16;int kv_mode=0;bool kv_q4=false,kv_int8=false,kv_hybrid=false,kv_rot=false;float*idx_tail=nullptr,*idx_dead=nullptr,*idx_pooled=nullptr;int32_t*idx_block_pos=nullptr;};
struct SessionState {int64_t layer_lo=0,layer_hi=24,max_cells=16,qsa_ord0=1,qsa_alloc=1,gdn_ord0=0,gdn_alloc=1;float*gdn_state=nullptr,*ple_hist=nullptr;QsaState*qsa_states=nullptr;int32_t ple_prev[2]={-1,-1};};
inline size_t owned_qsa(const SessionState&ss){return size_t(ss.qsa_alloc);}inline const QsaState&owned(const SessionState&ss,size_t j){return ss.qsa_states[size_t(ss.qsa_ord0)+j];}
inline bool conversation_session_sizes(const ModelGeometry&,const SessionState&ss,ConversationStateSizes&z,std::string&){z={};return ss.qsa_alloc==1&&ss.gdn_alloc==1;}
inline bool conversation_checkpoint_validate(const ConversationCheckpoint&c,const SessionState&,const ModelGeometry&,std::string&e){bool valid=c.gdn.size()==8&&c.ple.size()==4&&c.tails.size()==4&&c.dead.size()==4&&c.block_pos.size()==4;if(!valid)e="mock shape invalid";return valid;}
inline bool copy(void*dst,const void*src,size_t bytes,std::string&){if(bytes){++writes;std::memcpy(dst,src,bytes);}return true;}inline bool sync(std::string&){return dpct::get_current_device().queues_wait_and_throw()==0;}
inline uint64_t conversation_available_memory(){return uint64_t(1)<<40;}inline bool conversation_memory_admit(uint64_t,size_t,uint64_t){return memory_ok;}
inline size_t conversation_kv_bytes(const QsaState&,const ModelGeometry&,int64_t upto,bool){return size_t(upto+1)*4+64;}
inline bool conversation_kv_save(ConversationKv&image,const QsaState&ss,const ModelGeometry&,int64_t upto,bool,std::string&){size_t bytes=(size_t(upto/4)+1)*4;image.pooled.resize(bytes);return image.pooled.visit(0,bytes,[&](uint8_t*p,size_t n,size_t at){std::memcpy(p,reinterpret_cast<const uint8_t*>(ss.idx_pooled)+at,n);return true;});}
inline bool conversation_kv_restore(const ConversationKv&image,const QsaState&ss,const ModelGeometry&,int64_t,bool,std::string&e){if(device==restore_fail){e="mock transfer failure";return false;}++writes;return image.pooled.read(ss.idx_pooled,0,image.pooled.size());}
inline bool conversation_checkpoint_save(ConversationCheckpoint&c,const SessionState&ss,const ModelGeometry&,std::string&){++saves;c.gdn.assign(reinterpret_cast<uint8_t*>(ss.gdn_state),reinterpret_cast<uint8_t*>(ss.gdn_state)+8);c.ple.assign(reinterpret_cast<uint8_t*>(ss.ple_hist),reinterpret_cast<uint8_t*>(ss.ple_hist)+4);const auto&st=owned(ss,0);c.tails.assign(reinterpret_cast<uint8_t*>(st.idx_tail),reinterpret_cast<uint8_t*>(st.idx_tail)+4);c.dead.assign(reinterpret_cast<uint8_t*>(st.idx_dead),reinterpret_cast<uint8_t*>(st.idx_dead)+4);c.block_pos.assign(reinterpret_cast<uint8_t*>(st.idx_block_pos),reinterpret_cast<uint8_t*>(st.idx_block_pos)+4);return true;}
'''
POST=r'''
struct ConversationView {const std::vector<int32_t>&ids;const std::vector<ConversationImageKey>&images;const std::vector<ConversationCheckpoint>&checkpoints;bool cvec;};
inline bool conversation_snapshot_bytes(const ConversationView&view,const SessionState&ss,const ModelGeometry&g,const QsaState*,size_t&b,std::string&e){for(const auto&cp:view.checkpoints)if(!cp.stage_parts.empty()||!conversation_checkpoint_validate(cp,ss,g,e))return false;b=1024+batch_prefix::chain_bytes(view.checkpoints);return true;}
inline bool conversation_snapshot_save(SavedConversation&image,const ConversationView&view,const SessionState&ss,const ModelGeometry&g,const QsaState*,std::string&e){++parked_captures;image.live.ids=view.ids;image.live.imgs=view.images;if(!conversation_checkpoint_save(image.live,ss,g,e))return false;image.checkpoints=view.checkpoints;image.cvec=view.cvec;image.layer_lo=ss.layer_lo;image.layer_hi=ss.layer_hi;image.kv.resize(1);return conversation_kv_save(image.kv[0],owned(ss,0),g,int64_t(view.ids.size()),true,e);}
}
using ImgKey=strata::core::ConversationImageKey;
struct Owned {strata::core::SessionState ss;std::array<float,2>gdn;float ple,tail,dead;int32_t block;std::array<float,8>pooled,foreign;std::array<strata::core::QsaState,3>states;
 Owned(int stage){ss.layer_lo=stage?24:0;ss.layer_hi=stage?48:24;ss.gdn_ord0=stage;ss.qsa_ord0=stage?2:1;ss.gdn_state=gdn.data();ss.ple_hist=&ple;ss.qsa_states=states.data();auto&st=states[size_t(ss.qsa_ord0)];st.idx_tail=&tail;st.idx_dead=&dead;st.idx_pooled=pooled.data();st.idx_block_pos=&block;states[0].idx_pooled=foreign.data();reset(stage);}
 void reset(int stage){gdn={float(901+1000*stage),float(902+1000*stage)};ple=float(903+1000*stage);tail=float(904+1000*stage);dead=float(909+1000*stage);block=3;pooled.fill(float(999+1000*stage));pooled[0]=float(700+1000*stage);foreign.fill(444);ss.ple_prev[0]=11;ss.ple_prev[1]=12;}
};
struct Stage {int dev=1;strata::core::SessionState ss;};
struct Mtp {strata::core::QsaState state;bool idle(std::string&){return true;}auto&kv_state(){return state;}void set_prompt_len(int64_t){}};
struct Verify {bool wait_commit(std::string&){return true;}};
struct BSlot {uint64_t request_id=11,generation=7;bool active=true,cached=false,partial=false,img=false,cvec=true;std::vector<int32_t>ids;std::vector<ConvCheckpoint>checks;};
template<class T>std::vector<uint8_t>bytes(const std::vector<T>&v){const auto*p=reinterpret_cast<const uint8_t*>(v.data());return {p,p+v.size()*sizeof(T)};}
ConvCheckpoint part(int L,int offset){ConvCheckpoint c;for(int i=1;i<=L;++i)c.ids.push_back(i);c.gdn=bytes<float>({float(offset+1),float(offset+2)});c.ple=bytes<float>({float(offset+3)});c.tails=bytes<float>({float(offset+4)});c.dead=bytes<float>({float(offset+11)});c.block_pos=bytes<int32_t>({L%4});return c;}
ConvCheckpoint point(int L,int offset){auto c=part(L,offset);c.stage_parts.push_back(part(L,offset+100));c.used=L;c.pinned=L==7;return c;}
int main(int argc,char**argv){if(argc>1&&std::string(argv[1])=="config"){try{const auto&c=strata::core::batch_prefix::settings();std::cout<<"CONFIG enabled="<<c.enabled<<" chain="<<c.chain_bytes<<" transfer="<<c.transfer_bytes<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}using namespace strata::core;using namespace strata::core::batch_prefix;
 assert(settings().enabled);int rejected=0;std::string err;
 auto bad=[&](auto f){try{f();}catch(const std::exception&){++rejected;return;}assert(false);};auto reject=[&](auto f,bool unchanged=true){int before=writes;assert(!f());if(unchanged)assert(writes==before);++rejected;};
 bad([&]{startup_bytes(1,3,3);});bad([&]{startup_bytes(1,6,9);});bad([&]{mul(SIZE_MAX,2);});
 std::vector<int32_t> tokens;for(int i=1;i<=12;++i)tokens.push_back(i);std::vector<ConvCheckpoint>sample={point(5,100),point(7,200),point(11,300)},out;
 assert(clone(out,sample,tokens,11,1,3,0,1<<20,err)&&out.size()==3&&out[1].pinned);assert(clone(out,sample,tokens,7,1,3,0,1<<20,err)&&out.size()==2&&out[0].ids.size()==5&&out[1].pinned);
 auto invalid=sample;invalid[0].stage_parts.clear();reject([&]{return clone(out,invalid,tokens,11,1,3,0,1<<20,err);});invalid=sample;invalid[1].ids[0]=99;reject([&]{return clone(out,invalid,tokens,11,1,3,0,1<<20,err);});reject([&]{return clone(out,sample,tokens,11,1,2,0,1<<20,err);});reject([&]{return clone(out,sample,tokens,11,1,3,0,1,err);});assert(!eligible({11,7,true,false,false},false));assert(eligible({11,7,true,false,false},true));
 Owned main0(0),main1(1);SessionState&ss=main0.ss;std::vector<std::unique_ptr<Stage>>stages;stages.push_back(std::make_unique<Stage>());stages[0]->ss=main1.ss;
 std::vector<std::array<std::unique_ptr<Owned>,2>>owner(2);std::vector<std::vector<std::unique_ptr<SessionState>>>bslot_ss(2);for(int k=0;k<2;++k)for(int b=0;b<2;++b){owner[k][b]=std::make_unique<Owned>(k);bslot_ss[k].push_back(std::make_unique<SessionState>(owner[k][b]->ss));}
 std::vector<BSlot>bs(2);bs[0].ids=tokens;bs[0].checks=sample;bs[1].request_id=12;bs[1].generation=9;bs[1].ids=tokens;bs[1].checks=sample;
 std::vector<ConvCheckpoint>checks;std::vector<int64_t>cur;for(int i=1;i<=6;++i)cur.push_back(i);bool batch_chain_enabled=true;int batch_mtp=0;Mtp mtp;std::vector<std::unique_ptr<Mtp>>slot_mtp;ModelGeometry g,draft_geometry;Verify ver;ConversationCache conversations(1<<20,2);
 struct Options {int64_t prompt_cache=3,conversation_cache_min_free_mib=0;}o;
 std::function<size_t()>batch_slot_chain_bytes=[&](){size_t n=0;for(auto&s:bs)n=add(n,chain_bytes(s.checks));return n;};
'''
TEST=r'''
 auto original_checks=checks;int before=writes;assert(copy_from_slot(0,&bs[0].checks[0],err,11,7));assert(saves==0);assert(checks.size()==1&&checks[0].ids.size()==5);
 assert(main0.pooled[1]==111&&main1.pooled[1]==211);assert(main0.gdn[0]==101&&main1.gdn[0]==201&&main0.ple==103&&main1.ple==203&&main0.tail==104&&main1.tail==204&&main0.dead==111&&main1.dead==211&&main0.block==1&&main1.block==1);assert(ss.ple_prev[0]==4&&ss.ple_prev[1]==5);
 assert(owner[0][0]->gdn[0]==901&&owner[1][0]->gdn[0]==1901&&owner[0][1]->gdn[0]==901);assert(main0.foreign[0]==444&&main1.foreign[0]==444);assert(writes>before);
 // Preserve frozen original ordering as a numerical negative: selected dead111 is overwritten by donor spare999.
 batch_chain_enabled=false;main0.reset(0);main1.reset(1);assert(copy_from_slot(0,&bs[0].checks[0],err));assert(main0.pooled[1]==999&&main1.pooled[1]==1999);batch_chain_enabled=true;
 cur.clear();for(int i=1;i<=12;++i)cur.push_back(i);assert(copy_from_slot(0,&bs[0].checks.back(),err,11,7));assert(checks.size()==3&&checks.front().ids.size()==5&&checks[1].pinned&&checks.back().ids.size()==11);assert(main0.pooled[2]==311&&main1.pooled[2]==411);
 reject([&]{return copy_from_slot(0,nullptr,err,11,7);}); // Active live state must never be captured.
 reject([&]{return copy_from_slot(-1,nullptr,err,11,7);});reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,6);});
 auto foreign=bs[0].checks[0];reject([&]{return copy_from_slot(0,&foreign,err,11,7);});
 auto saved=bs[0].checks;bs[0].checks[0].stage_parts.clear();reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});bs[0].checks=saved;
 bs[0].checks[0].stage_parts[0].gdn.clear();reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});bs[0].checks=saved;
 auto oldcur=cur;cur[0]=99;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});cur=oldcur;
 bslot_ss[1][0]->layer_lo=25;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});bslot_ss[1][0]->layer_lo=24;
 bslot_ss[1][0]->qsa_states[2].kv_int8=true;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});bslot_ss[1][0]->qsa_states[2].kv_int8=false;
 memory_ok=false;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});memory_ok=true;
 changed=[&]{++bs[0].generation;};waits=0;change_at_wait=1;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);});bs[0].generation=7;change_at_wait=-1;
 // A device transfer failure can leave destination writes; it MUST NOT publish imported checks or advertise reuse.
 checks={point(3,900)};restore_fail=1;reject([&]{return copy_from_slot(0,&bs[0].checks[0],err,11,7);},false);assert(checks.size()==1&&checks[0].ids.size()==3);restore_fail=-1;
 // Idle live donor is allowed after quiescence, but source mutable recurrence capture is never used for active donor.
 bs[0].active=false;bs[0].cached=true;cur.push_back(13);int oldsave=saves;assert(copy_from_slot(0,nullptr,err,11,7));assert(saves==oldsave+2&&checks.size()==3);assert(main0.gdn[0]==901&&main1.gdn[0]==1901);
 // Admission parks an idle branch before actual target state overwrites, using all stage parts.
 bs[1].active=false;bs[1].cached=true;std::vector<int32_t>next=tokens;next.push_back(13);cur.assign(next.begin(),next.end());main0.gdn[0]=321;main1.gdn[0]=421;
 int captures=parked_captures;assert(copy_to_slot(1,next,err));assert(parked_captures==captures+2&&conversations.size()==1);assert(owner[0][1]->gdn[0]==321&&owner[1][1]->gdn[0]==421);
 std::vector<int64_t>oldprompt(tokens.begin(),tokens.end());oldprompt.push_back(99);auto parked=conversations.best(oldprompt,{},true);assert(parked.tokens==12&&parked.live);auto image=conversations.take(parked.index);float oldstate=0;std::memcpy(&oldstate,image.live.gdn.data(),4);assert(oldstate==901&&image.checkpoints.size()==3&&image.stage_images.size()==1&&image.stage_images[0].checkpoints.size()==3);
 bs[1].active=true;reject([&]{return copy_to_slot(1,next,err);});bs[1].active=false;auto goodchecks=checks;checks[0].stage_parts.clear();reject([&]{return copy_to_slot(1,next,err);});checks=goodchecks;
 auto wrong=next;wrong[0]=99;reject([&]{return copy_to_slot(1,wrong,err);});
 std::cout<<"PASS source-bound full chain root/pin/leaf+all-stage copy; original spare999/1999 vs corrected111/211; active source remains unchanged; controls="<<rejected<<"; no SDK/GPU/model qualification\n";
}
'''
with tempfile.TemporaryDirectory(prefix='batch-chain25-cpu-') as directory:
 w=Path(directory);base=Path(plan['compiled20_base']);baseplan=ROOT/plan['compiled20_plan'];assert sha(baseplan)==plan['compiled20_plan_sha256']
 for rel,h in json.loads(baseplan.read_text())['expected_patched_source_sha256'].items():assert sha(base/rel)==h
 shutil.copytree(base,w/'source');source=w/'source'
 for item in plan['patches']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];subprocess.run(['git','apply','--check',str(p)],cwd=source,check=True);subprocess.run(['git','apply',str(p)],cwd=source,check=True)
 for rel,h in plan['source_sha256'].items():assert sha(source/rel)==h
 state=(source/'sycl/src/core/conversation_state.cpp').read_text();restore=extract(state,'bool conversation_checkpoint_restore(')
 generate=(source/'sycl/src/program/generate.cpp').read_text()
 # Exact frozen public guards remain; batch cannot silently request public pin or fresh.
 assert 'o.batch>0 || use_mtp || o.pipeline_windows>0 || geni' in generate and 'req_fresh && (o.batch>0 ||' in generate
 assert 'req_ckpt && n>1 && !checkpoint_at(n-1)' in generate and 'active_checkpoint=%d' in generate
 functions=[extract(generate,'auto batch_chain_room='),extract(generate,'auto donor_identity='),extract(generate,'auto chain_clone='),extract(generate,'auto staged_checkpoint_valid='),extract(generate,'auto transfer_admit='),extract(generate,'auto park_slot='),extract(generate,'auto copy_to_slot ='),extract(generate,'auto copy_from_slot =')]
 preflight=extract(generate,'if(batch_chain_enabled) {')
 preflight_fn='int source_preflight(int cap,int asked,bool borrow){using namespace strata::core;Owned a(0),b(1);auto&ss=a.ss;std::vector<std::unique_ptr<Stage>>stages;stages.push_back(std::make_unique<Stage>());stages[0]->ss=b.ss;ModelGeometry g;std::string err;const bool batch_chain_enabled=true,strict_batch_identity=true,split_same=false;int requested_batch=asked;struct Opt{bool serve=true,adapt_async=false,no_prefill_borrow=true,kv_grow=false;int adapt_every=0,prompt_cache=3,max_context=16;}o;o.prompt_cache=cap;o.no_prefill_borrow=borrow;size_t batch_chain_point_bound=0;'+preflight+'return 0;}\n'
 post=POST.replace('int main(int argc,char**argv)',preflight_fn+'int main(int argc,char**argv)')
 post=post.replace('assert(settings().enabled);int rejected=0;','assert(settings().enabled);assert(source_preflight(3,2,true)==0);assert(source_preflight(1,2,true)==2);try{source_preflight(3,3,true);assert(false);}catch(const std::invalid_argument&){}assert(source_preflight(3,2,false)==2);int rejected=0;')
 cpp=PRE+restore+post+'\n'.join(functions)+TEST;(w/'test.cpp').write_text(cpp)
 command='g++ -std=c++20 -Wall -Wextra -Werror -Wno-misleading-indentation -fsanitize=address,undefined -fno-sanitize-recover=all -g -I/work/source/sycl/include -I/work/source/include /work/test.cpp -o /work/test && STRATA_BATCH_FULL_STATE_CHAIN=1 /work/test'
 result=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc',command],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 for key,value in [('STRATA_BATCH_FULL_STATE_CHAIN','2'),('STRATA_BATCH_CHAIN_MIB','0'),('STRATA_BATCH_CHAIN_MIB','-1'),('STRATA_BATCH_CHAIN_MIB','16385'),('STRATA_BATCH_CHAIN_MIB','8foo'),('STRATA_BATCH_TRANSFER_MIB','0'),('STRATA_BATCH_TRANSFER_MIB','1025')]:
  negative=subprocess.run(['docker','run','--rm','--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--entrypoint','/work/test','-v',str(w)+':/work','-e','STRATA_BATCH_FULL_STATE_CHAIN=1','-e',key+'='+value,'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','config'],capture_output=True,text=True)
  assert negative.returncode==2,(key,value,negative.stdout,negative.stderr)
 print('PASS7 malformed/unbounded cache configuration negatives')
 print(result.stdout.strip());print('PASS exact tracked source reconstruction + actual policy/extracted transfer/restore under host ASan/UBSan. No full TU/SDK/GPU/model data read.')
