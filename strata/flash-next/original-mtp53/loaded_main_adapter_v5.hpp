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
namespace strata::kernels::original_mtp53 {
// Diagnostic-only real MAIN integration; no alternate draft checkpoint or main math.
class LoadedMainDraft53 {
 sycl::queue& q;OwnedRuntime53 owner;StepWeights53 W{};StepScratch53 S{};
 std::unique_ptr<DraftQsaTransaction> state;std::unique_ptr<OriginalDraftStep53> draft;
 float* incoming=nullptr;float* committed_h=nullptr;int32_t* token=nullptr;int32_t* host_token=nullptr;bool failed=false;
 void check(){if(failed)throw std::runtime_error("MTP53 MAIN diagnostic failure latched");}
 static bool same_bytes(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size())==0);}
 void catchup(int32_t next,const float* hidden){check();*host_token=next;draft->prepare_candidate(next);q.memcpy(token,host_token,4);draft->record(token,hidden,W,S);q.wait_and_throw();draft->settle_fixture(true);q.wait_and_throw();}
 int32_t proposal(){std::vector<float> logits(248320);q.memcpy(logits.data(),S.logits,logits.size()*4).wait_and_throw();int32_t best=0;for(int32_t i=0;i<248320;++i){if(!std::isfinite(logits[i]))throw std::runtime_error("MTP53 own head nonfinite");if(logits[i]>logits[best])best=i;}return best;}
public:
 explicit LoadedMainDraft53(const char* path,sycl::queue& queue):q(queue),owner(q){
    strata::core::original_mtp53::TensorViews source;std::string error;if(!source.load(path,error))throw std::runtime_error(error);
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
    const int capacity=256,max_blocks=capacity/4+2;S.attention={fl(12288),fl(6144),fl(512),fl(512),fl(128),owner.take<uint16_t>(128),fl(128),fl(512),fl(max_blocks),owner.take<int32_t>(2051),fl(qsa_decode_attn_scratch_floats(2051,qsa_real_shapes())),fl(6144),fl(6144),fl(2560),q8(6144,1)};
    auto hc=[&]{return HcScratch53{fl(10240),fl(320),fl(10240),fl(10240),fl(2560),fl(4),q8(10240,1)};};S.after_attention=fl(10240);S.ffn_hc=hc();S.head_hc=hc();S.next_hidden=fl(10240);S.logits=fl(248320);S.write_weights=fl(4);S.write_products=fl(10240);S.head_q8=q8(2560,1);
    S.ffn={fl(512),owner.take<int32_t>(10),fl(10),owner.take<unsigned long long>(10),owner.take<int32_t>(11),owner.take<int32_t>(1),owner.take<int32_t>(10),owner.take<int32_t>(10),owner.take<unsigned char>(native_expert_scratch_bytes(10,640)),q8(2560,1),fl(25600),fl(1),fl(640),fl(640),q8(640,1),fl(2560),fl(2560),fl(2560)};
    auto bank=[&]{DraftBank B{owner.take<uint16_t>(capacity*512),owner.take<uint16_t>(capacity*512),fl(384),fl(128),fl(max_blocks*128),owner.take<int32_t>(1)};q.memset(B.key,0,capacity*512*2);q.memset(B.value,0,capacity*512*2);q.memset(B.tail,0,384*4);q.memset(B.dead,0,128*4);q.memset(B.pooled,0,max_blocks*128*4);q.memset(B.block_pos,0,4);return B;};DraftBank committed=bank(),working=bank();
    auto* host_step=owner.host<int32_t>(4);auto* host_positions=owner.host<int32_t>(24);auto* step=owner.take<int32_t>(4);auto* pos=owner.take<int32_t>(24);auto* pages=owner.take<int32_t>(capacity/4);std::vector<int32_t> pg(capacity/4);for(size_t i=0;i<pg.size();++i)pg[i]=int(i);q.memcpy(pages,pg.data(),pg.size()*4);state.reset(new DraftQsaTransaction(committed,working,capacity,host_step,host_positions,step,pos,pages));
    incoming=fl(10240);committed_h=fl(10240);q.memset(committed_h,0,10240*4);token=owner.take<int32_t>(1);host_token=owner.host<int32_t>(1);draft.reset(new OriginalDraftStep53(*state,committed_h,S.next_hidden,&q));q.wait_and_throw();

 }
 void advance_actual(int32_t next,const float* target_hidden){try{catchup(next,target_hidden);}catch(...){failed=true;throw;}}
 int32_t propose_actual(){check();return proposal();}
 int64_t cells()const{return state->committed_cells();}
 bool verify_actual(strata::core::Verifier& main,int32_t input,int32_t guess,int64_t position,strata::core::PoolMultiFn pool,void* user,int remaining,std::array<int32_t,2>& actual,bool& accepted,std::string& error){
  check();const bool ok=draft->verify_bounded_and_catchup(main,input,guess,position,pool,user,actual,token,host_token,W,S,remaining,accepted,error);if(!ok)failed=true;return ok;
 }
 bool exercise(strata::core::Verifier& main,strata::core::SessionState& ss,const strata::core::ModelGeometry& geometry,strata::core::PoolMultiFn pool,void* user)try{
  check();std::string error;int32_t input=248045,next=-1;
  const char* self_commit=std::getenv("STRATA_ONE_TOKEN_COMMIT");
  if(!self_commit || std::strcmp(self_commit,"0")!=0)throw std::runtime_error("MTP53 control requires exact ONE_TOKEN_COMMIT=0");
  std::vector<int32_t> target_history{input};
  if(!main.run(1,&input,0,pool,user,&next,error)||!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);
  if(!main.final_R_all())throw std::runtime_error("MTP53 MAIN hidden missing");
  catchup(next,main.final_R_all());int64_t position=1;input=next;
  // The uncommitted T1 window is the unchanged target's independent greedy control.
  // T2 re-evaluates the same committed prefix; force-wrong is explicitly a negative control.
  for(int trial=0;trial<3;++trial){
   // A run without commit still advances PLE history and writes speculative QSA cells.
   // Save original running state AND the touched positional pages through unchanged SDK APIs.
   if(target_history.size()!=size_t(position) || ss.qsa_ord0!=0 || ss.qsa_alloc!=geometry.n_qsa_layers())throw std::runtime_error("MTP53 control snapshot prefix/carve differs");
   dpct::get_current_device().queues_wait_and_throw();
   strata::core::ConversationCheckpoint checkpoint;checkpoint.ids=target_history;
   if(!strata::core::conversation_checkpoint_save(checkpoint,ss,geometry,error))throw std::runtime_error(error);
   const int64_t touched=position+2;
   std::vector<strata::core::ConversationKv> kv(size_t(ss.qsa_alloc));
   for(size_t layer=0;layer<kv.size();++layer)
    if(!strata::core::conversation_kv_save(kv[layer],ss.qsa_states[layer],geometry,touched,true,error))throw std::runtime_error(error);
   int32_t expected=-1;const bool control_ok=main.run(1,&input,position,pool,user,&expected,error);
   const std::string control_error=error;
   dpct::get_current_device().queues_wait_and_throw();
   // Restore running state first: its spare pooled-row reconstruction is then overwritten
   // by the exact original authoritative K/V/pooled byte image, including touched pages.
   if(!strata::core::conversation_checkpoint_restore(checkpoint,ss,geometry,error))throw std::runtime_error(error);
   for(size_t layer=0;layer<kv.size();++layer)
    if(!strata::core::conversation_kv_restore(kv[layer],ss.qsa_states[layer],geometry,touched,true,error))throw std::runtime_error(error);
   dpct::get_current_device().queues_wait_and_throw();
   strata::core::ConversationCheckpoint restored;restored.ids=target_history;
   if(!strata::core::conversation_checkpoint_save(restored,ss,geometry,error))throw std::runtime_error(error);
   if(!same_bytes(restored.gdn,checkpoint.gdn) || !same_bytes(restored.ple,checkpoint.ple) || !same_bytes(restored.tails,checkpoint.tails) || !same_bytes(restored.dead,checkpoint.dead) || !same_bytes(restored.block_pos,checkpoint.block_pos) || ss.ple_prev[0]!=(position>=2?target_history[size_t(position)-2]:-1) || ss.ple_prev[1]!=target_history.back())throw std::runtime_error("MTP53 control running state did not restore exactly");
   for(size_t layer=0;layer<kv.size();++layer){uint64_t fingerprint=0;
    if(!strata::core::conversation_kv_verify(kv[layer],ss.qsa_states[layer],geometry,touched,true,fingerprint,error))throw std::runtime_error(error);
   }
   std::printf("MTP53_MAIN_CONTROL_RESTORED trial=%d base_position=%lld layers=%zu touched_cells=%lld GDN_PLE_indexer_KV_exact=1\n",trial,(long long)position,kv.size(),(long long)touched);std::fflush(stdout);
   if(!control_ok)throw std::runtime_error(control_error);

   const int32_t proposed=proposal();const int32_t guess=trial==0?(expected+1)%248320:trial==1?expected:proposed;
   std::array<int32_t,2> actual{};bool accepted=false;
   if(!draft->verify_and_catchup(main,input,guess,position,pool,user,actual,token,host_token,W,S,accepted,error))throw std::runtime_error(error);
   std::printf("MTP53_MAIN_PREASSERT trial=%d role=%s base_position=%lld input=%d expected=%d own_proposal=%d guess=%d accepted=%d actual0=%d actual1=%d draft_cells=%d greedy_equal=%d control_acceptance_equal=%d\n",trial,trial==0?"forced_reject":trial==1?"forced_accept":"own_proposal",(long long)position,input,expected,proposed,guess,int(accepted),actual[0],actual[1],state->committed_cells(),int(actual[0]==expected),int((trial!=0 || !accepted) && (trial!=1 || accepted)));std::fflush(stdout);
   if(actual[0]!=expected || (trial==0&&accepted) || (trial==1&&!accepted))throw std::runtime_error("MTP53 unchanged target greedy/negative control differed");
   const int kept=accepted?2:1;target_history.push_back(input);if(accepted)target_history.push_back(guess);position+=kept;input=actual[kept-1];
   if(state->committed_cells()!=position)throw std::runtime_error("MTP53 actual MAIN/draft catch-up position differs");
   std::printf("MTP53_MAIN_STEP trial=%d role=%s expected=%d own_proposal=%d guess=%d accepted=%d actual0=%d actual1=%d main_position=%lld draft_cells=%d\n",trial,trial==0?"forced_reject":trial==1?"forced_accept":"own_proposal",expected,proposed,guess,int(accepted),actual[0],actual[1],(long long)position,state->committed_cells());std::fflush(stdout);
  }
  owner.retire();std::printf("MTP53_MAIN_DIAGNOSTIC_PASS target_greedy_first_pick_equal=1 real_main_verification=1 trials=3 original_draft_tensors=34 full64_equivalence=0 numerical_draft_qualified=0 speed_qualified=0\n");return true;
 }catch(const std::exception& e){failed=true;std::fprintf(stderr,"MTP53_MAIN_DIAGNOSTIC_ERROR %s\n",e.what());return false;}
};
}
