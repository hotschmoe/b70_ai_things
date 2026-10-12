#pragma once
#include "strata/kernels/original_mtp53_draft_step.hpp"
#include "strata/core/original_mtp53_owned_runtime.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <dpct/dpct.hpp>
#include <map>
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
#include "strata/core/on_device.hpp"
namespace strata::kernels::original_mtp53 {
// Two-stage diagnostic, own publisher tensors on LAST GPU; no shared draft/main weights.
struct TargetStage53 {strata::core::Verifier* verifier;strata::core::SessionState* session;int device,lb,le;};
class LoadedMainDraft53 {
 sycl::queue& q;int device_=1,capacity_=4096;std::unique_ptr<strata::core::original_mtp53::TensorViews> source_keep_;OwnedRuntime53 owner;
 strata::core::Verifier* target_=nullptr;strata::core::Verifier* last_=nullptr;const float* target_rows_=nullptr;const float* prompt_rows_=nullptr;int prompt_count_=0;StepWeights53 W{};StepScratch53 S{};
 std::unique_ptr<DraftQsaTransaction> state;std::unique_ptr<OriginalDraftStep53> draft;
 float* incoming=nullptr;float* committed_h=nullptr;int32_t* token=nullptr;int32_t* host_token=nullptr;bool failed=false;PhaseTiming53* timing_=nullptr;
 void check(){if(failed)throw std::runtime_error("MTP53 MAIN diagnostic failure latched");}
 static bool same_bytes(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size())==0);}
 void catchup(int32_t next,const float* hidden){check();check_hidden(hidden);{PhaseScope53 phase(timing_,"draft_prepare_submit");*host_token=next;draft->prepare_candidate(next);q.memcpy(token,host_token,4);}{PhaseScope53 phase(timing_,"draft_record_wait");draft->record(token,hidden,W,S);q.wait_and_throw();}{PhaseScope53 phase(timing_,"draft_settle_wait");draft->settle_fixture(true);q.wait_and_throw();}}
 int32_t proposal(){std::vector<float> logits(248320);q.memcpy(logits.data(),S.logits,logits.size()*4).wait_and_throw();int32_t best=0;for(int32_t i=0;i<248320;++i){if(!std::isfinite(logits[i]))throw std::runtime_error("MTP53 own head nonfinite");if(logits[i]>logits[best])best=i;}return best;}
public:
 explicit LoadedMainDraft53(const char* path,sycl::queue& queue,int owner_device,int context_capacity):q(queue),device_(owner_device),capacity_(context_capacity),source_keep_(new strata::core::original_mtp53::TensorViews),owner(q){
    const strata::core::OnDevice on(device_);
    if(device_!=1 || capacity_!=4096)throw std::runtime_error("pair53 exact LAST device1/context4096 required");
    auto& source=*source_keep_;std::string error;if(!source.load(path,error))throw std::runtime_error(error);
    if(!q.get_device().is_gpu() || !q.is_in_order())throw std::runtime_error("owned ordered GPU queue required");
    std::map<std::string,const void*> device;
    for(const auto& spec:strata::core::original_mtp53::specs())if(std::string(spec.name).find("_exps.weight")==std::string::npos)device[spec.name]=owner.upload(source,spec.name);
    // Lossless per-expert concatenation only: original blocks untouched, no Q2_0/requantization.
    const auto layout=native_expert_layout(12,8,2560,640);if(layout.up_off!=921600 || layout.down_off!=1843200 || layout.bytes!=3584000)throw std::runtime_error("original expert layout mismatch");
    auto* experts=owner.take<unsigned char>(layout.bytes*512);const auto gu=source.at("blk.48.ffn_gate_exps.weight"),up=source.at("blk.48.ffn_up_exps.weight"),down=source.at("blk.48.ffn_down_exps.weight");
    for(int e=0;e<512;++e){q.memcpy(experts+e*layout.bytes,static_cast<const unsigned char*>(gu.readonly_gguf_bytes)+e*layout.up_off,layout.up_off);q.memcpy(experts+e*layout.bytes+layout.up_off,static_cast<const unsigned char*>(up.readonly_gguf_bytes)+e*layout.up_off,layout.up_off);q.memcpy(experts+e*layout.bytes+layout.down_off,static_cast<const unsigned char*>(down.readonly_gguf_bytes)+e*(layout.bytes-layout.down_off),layout.bytes-layout.down_off);}
    auto w=[&](const char* n){return device.at(n);};auto f=[&](const char* n){return static_cast<const float*>(w(n));};auto b=[&](const char* n){return static_cast<const uint16_t*>(w(n));};
    W.front={w("token_embd.weight"),f("blk.48.nextn.enorm.weight"),f("blk.48.nextn.hnorm.weight"),w("blk.48.nextn.eh_proj.weight"),f("blk.48.hc_attn_norm.weight"),w("blk.48.hc_attn_down.weight"),w("blk.48.hc_attn_up.weight"),w("blk.48.hc_attn_inject.weight")};
    W.attention={w("blk.48.attn_q.weight"),w("blk.48.attn_k.weight"),w("blk.48.attn_v.weight"),w("blk.48.attn_output.weight"),f("blk.48.attn_q_norm.weight"),f("blk.48.attn_k_norm.weight"),b("blk.48.indexer.q_proj.weight"),b("blk.48.indexer.k_proj.weight"),f("blk.48.indexer.q_norm.weight"),f("blk.48.indexer.k_norm.weight")};
    W.ffn_hc={f("blk.48.hc_ffn_norm.weight"),14,w("blk.48.hc_ffn_down.weight"),w("blk.48.hc_ffn_up.weight"),w("blk.48.hc_ffn_inject.weight")};W.ffn={f("blk.48.ffn_gate_inp.weight"),f("blk.48.ffn_gate_inp_shexp.weight"),experts,w("blk.48.ffn_gate_shexp.weight"),w("blk.48.ffn_up_shexp.weight"),w("blk.48.ffn_down_shexp.weight")};W.head_hc={f("blk.48.nextn.hc_head_norm.weight"),12,w("blk.48.nextn.hc_head_down.weight"),w("blk.48.nextn.hc_head_up.weight"),nullptr};W.own_head_q6=w("output.weight");
    auto fl=[&](size_t n){return owner.take<float>(n);};auto q8=[&](int n,int t){return owner.take<unsigned char>(native_q8_1_bytes(n,t));};
    S.front={fl(2560),fl(2560),fl(10240),fl(20480),fl(10240),fl(10240),fl(320),fl(10240),fl(10240),fl(2560),fl(4),q8(5120,4)};
    const int capacity=capacity_,max_blocks=capacity/4+2;S.attention={fl(12288),fl(6144),fl(512),fl(512),fl(128),owner.take<uint16_t>(128),fl(128),fl(512),fl(max_blocks),owner.take<int32_t>(2051),fl(qsa_decode_attn_scratch_floats(2051,qsa_real_shapes())),fl(6144),fl(6144),fl(2560),q8(6144,1)};
    auto hc=[&]{return HcScratch53{fl(10240),fl(320),fl(10240),fl(10240),fl(2560),fl(4),q8(10240,1)};};S.after_attention=fl(10240);S.ffn_hc=hc();S.head_hc=hc();S.next_hidden=fl(10240);S.logits=fl(248320);S.write_weights=fl(4);S.write_products=fl(10240);S.head_q8=q8(2560,1);
    S.ffn={fl(512),owner.take<int32_t>(10),fl(10),owner.take<unsigned long long>(10),owner.take<int32_t>(11),owner.take<int32_t>(1),owner.take<int32_t>(10),owner.take<int32_t>(10),owner.take<unsigned char>(native_expert_scratch_bytes(10,640)),q8(2560,1),fl(25600),fl(1),fl(640),fl(640),q8(640,1),fl(2560),fl(2560),fl(2560)};
    auto bank=[&]{DraftBank B{owner.take<uint16_t>(capacity*512),owner.take<uint16_t>(capacity*512),fl(384),fl(128),fl(max_blocks*128),owner.take<int32_t>(1)};q.memset(B.key,0,capacity*512*2);q.memset(B.value,0,capacity*512*2);q.memset(B.tail,0,384*4);q.memset(B.dead,0,128*4);q.memset(B.pooled,0,max_blocks*128*4);q.memset(B.block_pos,0,4);return B;};DraftBank committed=bank(),working=bank();
    auto* host_step=owner.host<int32_t>(4);auto* host_positions=owner.host<int32_t>(24);auto* step=owner.take<int32_t>(4);auto* pos=owner.take<int32_t>(24);auto* pages=owner.take<int32_t>(capacity/4);auto* pg=owner.host<int32_t>(size_t(capacity/4));for(int i=0;i<capacity/4;++i)pg[i]=i;q.memcpy(pages,pg,size_t(capacity/4)*4);state.reset(new DraftQsaTransaction(committed,working,capacity,host_step,host_positions,step,pos,pages));
    incoming=fl(10240);committed_h=fl(10240);q.memset(committed_h,0,10240*4);token=owner.take<int32_t>(1);host_token=owner.host<int32_t>(1);draft.reset(new OriginalDraftStep53(*state,committed_h,S.next_hidden,&q));q.wait_and_throw();source_keep_.reset();

 }
 ~LoadedMainDraft53(){const strata::core::OnDevice on(device_);owner.retire();}
 void set_timing(PhaseTiming53* p){const strata::core::OnDevice on(device_);timing_=p;draft->set_timing(p);}
 void advance_actual(int32_t next,const float* target_hidden){const strata::core::OnDevice on(device_);try{catchup(next,target_hidden);}catch(...){failed=true;throw;}}
 int32_t propose_actual(){const strata::core::OnDevice on(device_);check();PhaseScope53 phase(timing_,"own_head_read_argmax");return proposal();}
 int64_t cells()const{return state->committed_cells();}
 bool verify_actual(strata::core::Verifier& main,int32_t input,int32_t guess,int64_t position,strata::core::PoolMultiFn pool,void* user,int remaining,std::array<int32_t,2>& actual,bool& accepted,std::string& error){
  const strata::core::OnDevice on(device_);check();if(&main!=target_ || main.final_R_all()!=target_rows_)throw std::runtime_error("pair53 target owner changed");const bool ok=draft->verify_bounded_and_catchup(main,input,guess,position,pool,user,actual,token,host_token,W,S,remaining,accepted,error);if(!ok)failed=true;return ok;
 }

 void bind_target(strata::core::Verifier& main,strata::core::Verifier& last){
  const strata::core::OnDevice on(device_);check();
  if(&main==&last || main.device()!=0 || last.device()!=device_ || !last.stream() || last.stream()->get_context()!=q.get_context() || last.stream()->get_device()!=q.get_device() || !main.final_R_all())throw std::runtime_error("pair53 exact target LAST queue/context required");
  const float* rows=main.final_R_all();if(sycl::get_pointer_type(rows,q.get_context())!=sycl::usm::alloc::device || sycl::get_pointer_device(rows,q.get_context())!=q.get_device())throw std::runtime_error("pair53 target residual allocation not owned on LAST GPU");
  target_=&main;last_=&last;target_rows_=rows;
 }
 void check_hidden(const float* hidden){
  if(!target_ || !last_ || target_->final_R_all()!=target_rows_ || !hidden)throw std::runtime_error("pair53 target not bound/current");
  const uintptr_t h=reinterpret_cast<uintptr_t>(hidden),t=reinterpret_cast<uintptr_t>(target_rows_),p=reinterpret_cast<uintptr_t>(prompt_rows_);
  const size_t stride=10240*4;
  const bool target=h>=t && h-t<2*stride && (h-t)%stride==0;
  const bool prompt=prompt_rows_ && h>=p && h-p<size_t(prompt_count_)*stride && (h-p)%stride==0;
  if(!target && !prompt)throw std::runtime_error("pair53 hidden outside validated target/prompt rows");
 }
 bool prefill_chunk(const float* rows,const std::vector<int64_t>& prompt,int T,int64_t pos,std::string& error)try{
  const strata::core::OnDevice on(device_);check();
  if(T<1 || T>512 || pos!=cells() || pos+T>=int64_t(prompt.size()) || !rows || sycl::get_pointer_type(rows,q.get_context())!=sycl::usm::alloc::device || sycl::get_pointer_device(rows,q.get_context())!=q.get_device())throw std::runtime_error("pair53 actual LAST prefill source/position/context differs");
  prompt_rows_=rows;prompt_count_=T;
  try{for(int t=0;t<T;++t){const int64_t next=prompt[size_t(pos+t+1)];if(next<0 || next>=248320)throw std::runtime_error("pair53 prompt next token differs");advance_actual(int32_t(next),rows+size_t(t)*10240);}}
  catch(...){prompt_rows_=nullptr;prompt_count_=0;throw;}
  prompt_rows_=nullptr;prompt_count_=0;
  if(cells()!=pos+T)throw std::runtime_error("pair53 original prompt recurrence count differs");return true;
 }catch(const std::exception& e){failed=true;error=e.what();return false;}
 bool exercise_pair(strata::core::Verifier& main,const std::vector<TargetStage53>& stages,const strata::core::ModelGeometry& g,strata::core::PoolMultiFn pool,void* user)try{
  const strata::core::OnDevice on(device_);check();
  if(stages.size()!=2 || stages[0].verifier!=&main || stages[1].verifier!=last_ || stages[0].device!=0 || stages[1].device!=1 || stages[0].lb!=0 || stages[0].le!=24 || stages[1].lb!=24 || stages[1].le!=48 || cells()!=0)throw std::runtime_error("pair53 forced control owner roster/initial state differs");
  for(const auto& st:stages)if(!st.session || st.session->layer_lo!=st.lb || st.session->layer_hi!=st.le || st.verifier->device()!=st.device)throw std::runtime_error("pair53 stage/session carve differs");
  std::string error;int32_t input=248045,next=-1;const char* explicit_commit=std::getenv("STRATA_ONE_TOKEN_COMMIT");if(!explicit_commit || std::strcmp(explicit_commit,"0")!=0)throw std::runtime_error("pair53 explicit commit required");
  if(!main.run(1,&input,0,pool,user,&next,error) || !main.commit(1,error) || !main.wait_commit(error))throw std::runtime_error(error);
  advance_actual(next,main.final_R_all());std::vector<int32_t> history{input};int64_t position=1;input=next;
  for(int trial=0;trial<3;++trial){
   const int64_t touched=position+2;std::vector<strata::core::ConversationCheckpoint> checkpoints(2);std::vector<std::vector<strata::core::ConversationKv>> images(2);std::array<std::array<int32_t,2>,2> saved_previous{};
   for(size_t i=0;i<stages.size();++i){const auto& st=stages[i];const strata::core::OnDevice stage_on(st.device);st.verifier->stream()->wait_and_throw();checkpoints[i].ids=history;auto& ss=*st.session;saved_previous[i]={ss.ple_prev[0],ss.ple_prev[1]};
    if(!strata::core::conversation_checkpoint_save(checkpoints[i],ss,g,error))throw std::runtime_error(error);
    images[i].resize(size_t(ss.qsa_alloc));for(size_t j=0;j<images[i].size();++j)if(!strata::core::conversation_kv_save(images[i][j],ss.qsa_states[size_t(ss.qsa_ord0)+j],g,touched,true,error))throw std::runtime_error(error);
   }
   int32_t expected=-1;const bool ok=main.run(1,&input,position,pool,user,&expected,error);const std::string original_error=error;
   for(size_t i=0;i<stages.size();++i){const auto& st=stages[i];const strata::core::OnDevice stage_on(st.device);st.verifier->stream()->wait_and_throw();auto& ss=*st.session;
    if(!strata::core::conversation_checkpoint_restore(checkpoints[i],ss,g,error))throw std::runtime_error(error);
    // Legacy restore seeds history on non-PLE stages too; retain the actual saved metadata exactly.
    ss.ple_prev[0]=saved_previous[i][0];ss.ple_prev[1]=saved_previous[i][1];
    for(size_t j=0;j<images[i].size();++j)if(!strata::core::conversation_kv_restore(images[i][j],ss.qsa_states[size_t(ss.qsa_ord0)+j],g,touched,true,error))throw std::runtime_error(error);
    st.verifier->stream()->wait_and_throw();strata::core::ConversationCheckpoint restored;restored.ids=history;if(!strata::core::conversation_checkpoint_save(restored,ss,g,error))throw std::runtime_error(error);
    if(!same_bytes(restored.gdn,checkpoints[i].gdn)||!same_bytes(restored.ple,checkpoints[i].ple)||!same_bytes(restored.tails,checkpoints[i].tails)||!same_bytes(restored.dead,checkpoints[i].dead)||!same_bytes(restored.block_pos,checkpoints[i].block_pos))throw std::runtime_error("pair53 target stage running state not exactly restored");
    if(ss.ple_prev[0]!=saved_previous[i][0] || ss.ple_prev[1]!=saved_previous[i][1])throw std::runtime_error("pair53 stage PLE metadata not restored");
    if(st.lb<=1 && st.le>1 && (ss.ple_prev[0]!=(position>=2?history[size_t(position)-2]:-1)||ss.ple_prev[1]!=history.back()))throw std::runtime_error("pair53 actual PLE owner history differs");
    for(size_t j=0;j<images[i].size();++j){uint64_t fingerprint=0;if(!strata::core::conversation_kv_verify(images[i][j],ss.qsa_states[size_t(ss.qsa_ord0)+j],g,touched,true,fingerprint,error))throw std::runtime_error(error);}
    std::printf("MTP53_PAIR_CONTROL_RESTORED trial=%d stage=%zu device=%d lb=%d le=%d base_position=%lld touched_cells=%lld state_KV_exact=1\n",trial,i,st.device,st.lb,st.le,(long long)position,(long long)touched);std::fflush(stdout);
   }
   if(!ok)throw std::runtime_error(original_error);
   const int32_t own=propose_actual(),guess=trial==0?(expected+1)%248320:trial==1?expected:own;std::array<int32_t,2> actual{};bool accepted=false;
   if(!verify_actual(main,input,guess,position,pool,user,64,actual,accepted,error))throw std::runtime_error(error);
   if(actual[0]!=expected || (trial==0&&accepted) || (trial==1&&!accepted))throw std::runtime_error("pair53 forced target greedy/acceptance differs");
   const bool eos=actual[0]==248044||actual[0]==248046;const int keep=accepted&&!eos?2:1;history.push_back(input);if(keep==2)history.push_back(guess);position+=keep;input=actual[keep-1];if(cells()!=position)throw std::runtime_error("pair53 draft catchup count differs");
   std::printf("MTP53_PAIR_CONTROL_STEP trial=%d accepted=%d committed=%d position=%lld target_greedy_equal=1\n",trial,int(accepted),keep,(long long)position);std::fflush(stdout);
  }
  return true;
 }catch(const std::exception& e){failed=true;std::fprintf(stderr,"MTP53_PAIR_CONTROL_ERROR %s\n",e.what());return false;}
};
}
