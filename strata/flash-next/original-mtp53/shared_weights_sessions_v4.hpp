#pragma once
// New source-only ownership split. No serving integration or legacy KV serialization.
#include "strata/kernels/original_mtp53_draft_step.hpp"
#include "strata/core/original_mtp53_owned_runtime.hpp"
#include "strata/core/on_device.hpp"
#include <dpct/dpct.hpp>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <limits.h>
#include <cmath>
namespace strata::kernels::original_mtp53 {
struct OriginalDraftIdentity53 {
 std::string canonical_path, publisher_sha256;
 uint64_t device,inode,size;int64_t mtime_ns,ctime_ns;
 void validate(const char* path)const {
  if(!path || canonical_path!=path || canonical_path.empty() || canonical_path[0]!='/' ||
     publisher_sha256!="8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231" || size!=2786204800ull)
   throw std::runtime_error("original shared34 identity/path differs");
  char resolved[PATH_MAX];struct stat leaf{};if(!::realpath(path,resolved) || canonical_path!=resolved || ::lstat(path,&leaf)!=0 || !S_ISREG(leaf.st_mode))throw std::runtime_error("original shared34 nonalias path required");
  struct stat st{};if(::stat(path,&st)!=0 || !S_ISREG(st.st_mode) || uint64_t(st.st_dev)!=device || uint64_t(st.st_ino)!=inode || uint64_t(st.st_size)!=size ||
     int64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec!=mtime_ns || int64_t(st.st_ctim.tv_sec)*1000000000+st.st_ctim.tv_nsec!=ctime_ns)
   throw std::runtime_error("original shared34 current file identity changed");
  // Hash is a required caller-owned fresh whole-file proof, not a hash computed here.
 }
};
class OriginalDraftWeights53 final {
 // Destruction order: USM owner drains/frees BEFORE the read-only GGUF unmaps;
 // the copied queue keeps its native implementation alive until both are gone.
 std::shared_ptr<sycl::queue> queue_keep_;
 std::unique_ptr<strata::core::original_mtp53::TensorViews> source_keep_;
 OwnedRuntime53 owner;StepWeights53 W{};const OriginalDraftIdentity53 identity_;
public:
 OriginalDraftWeights53(const char* path,const OriginalDraftIdentity53& identity,const sycl::queue& last_queue,int last_device)
 :queue_keep_(std::make_shared<sycl::queue>(last_queue)),source_keep_(new strata::core::original_mtp53::TensorViews),owner(*queue_keep_),identity_(identity){
  const strata::core::OnDevice on(last_device);identity_.validate(path);
  auto& q=*queue_keep_;if(last_device!=1 || q.get_device()!=dpct::get_in_order_queue().get_device() || q.get_context()!=dpct::get_in_order_queue().get_context() || !q.get_device().is_gpu() || !q.is_in_order())throw std::runtime_error("original shared34 LAST GPU1 ordered queue required");
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
    q.wait_and_throw();identity_.validate(path);
 }
 ~OriginalDraftWeights53(){const strata::core::OnDevice on(1);owner.retire();}
 OriginalDraftWeights53(const OriginalDraftWeights53&)=delete;
 OriginalDraftWeights53& operator=(const OriginalDraftWeights53&)=delete;
 const StepWeights53& weights()const noexcept{return W;}
 const OriginalDraftIdentity53& identity()const noexcept{return identity_;}
 sycl::queue& queue()const noexcept{return *queue_keep_;}
 void require_last(const sycl::queue& last,int device)const {
  if(device!=1 || last.get_context()!=queue().get_context() || last.get_device()!=queue().get_device() || !last.is_in_order())throw std::runtime_error("original shared34 LAST context/device changed");
  identity_.validate(identity_.canonical_path.c_str());
 }
};
struct OriginalDraftSlotOwner53 {uint64_t rid,generation;};
struct OriginalDraftPrivateSnapshot53 {
 int64_t cells;std::vector<uint8_t> committed,working,hidden;std::vector<float> logits;
 std::vector<uintptr_t> private_allocations;
};
class OriginalDraftSession53 final {
 // Shared pack lives longer than private buffers/graphs and queue references.
 std::shared_ptr<const OriginalDraftWeights53> weights_keep_;
 sycl::queue& q;OwnedRuntime53 owner;int capacity_=4096;OriginalDraftSlotOwner53 slot_;
 StepScratch53 S{};DraftBank committed_bank_{},working_bank_{};std::unique_ptr<DraftQsaTransaction> state;std::unique_ptr<OriginalDraftStep53> draft;
 float* incoming=nullptr;float* committed_h=nullptr;int32_t* token=nullptr;int32_t* host_token=nullptr;
 int32_t* host_step_keep_=nullptr;int32_t* host_positions_keep_=nullptr;int32_t* step_keep_=nullptr;int32_t* positions_keep_=nullptr;int32_t* pages_keep_=nullptr;
 bool failed=false,active_=false,has_logits_=false;
 static sycl::queue& require_pack(const std::shared_ptr<const OriginalDraftWeights53>& p){if(!p)throw std::invalid_argument("original shared34 owner required");return p->queue();}
 void check(OriginalDraftSlotOwner53 who)const {if(failed || who.rid!=slot_.rid || who.generation!=slot_.generation)throw std::runtime_error("original private slot failure/stale RID-generation");}
public:
 OriginalDraftSession53(std::shared_ptr<const OriginalDraftWeights53> pack,OriginalDraftSlotOwner53 who,const sycl::queue& last,int device)
 :weights_keep_(std::move(pack)),q(require_pack(weights_keep_)),owner(q),slot_(who){
  const strata::core::OnDevice on(device);if(!slot_.rid || !slot_.generation)throw std::invalid_argument("original private slot identity required");weights_keep_->require_last(last,device);
    auto fl=[&](size_t n){return owner.take<float>(n);};auto q8=[&](int n,int t){return owner.take<unsigned char>(native_q8_1_bytes(n,t));};
    S.front={fl(2560),fl(2560),fl(10240),fl(20480),fl(10240),fl(10240),fl(320),fl(10240),fl(10240),fl(2560),fl(4),q8(5120,4)};
    const int capacity=capacity_,max_blocks=capacity/4+2;S.attention={fl(12288),fl(6144),fl(512),fl(512),fl(128),owner.take<uint16_t>(128),fl(128),fl(512),fl(max_blocks),owner.take<int32_t>(2051),fl(qsa_decode_attn_scratch_floats(2051,qsa_real_shapes())),fl(6144),fl(6144),fl(2560),q8(6144,1)};
    auto hc=[&]{return HcScratch53{fl(10240),fl(320),fl(10240),fl(10240),fl(2560),fl(4),q8(10240,1)};};S.after_attention=fl(10240);S.ffn_hc=hc();S.head_hc=hc();S.next_hidden=fl(10240);S.logits=fl(248320);S.write_weights=fl(4);S.write_products=fl(10240);S.head_q8=q8(2560,1);
    S.ffn={fl(512),owner.take<int32_t>(10),fl(10),owner.take<unsigned long long>(10),owner.take<int32_t>(11),owner.take<int32_t>(1),owner.take<int32_t>(10),owner.take<int32_t>(10),owner.take<unsigned char>(native_expert_scratch_bytes(10,640)),q8(2560,1),fl(25600),fl(1),fl(640),fl(640),q8(640,1),fl(2560),fl(2560),fl(2560)};
    auto bank=[&]{DraftBank B{owner.take<uint16_t>(capacity*512),owner.take<uint16_t>(capacity*512),fl(384),fl(128),fl(max_blocks*128),owner.take<int32_t>(1)};q.memset(B.key,0,capacity*512*2);q.memset(B.value,0,capacity*512*2);q.memset(B.tail,0,384*4);q.memset(B.dead,0,128*4);q.memset(B.pooled,0,max_blocks*128*4);q.memset(B.block_pos,0,4);return B;};DraftBank committed=bank(),working=bank();committed_bank_=committed;working_bank_=working;
    auto* host_step=owner.host<int32_t>(4);auto* host_positions=owner.host<int32_t>(24);auto* step=owner.take<int32_t>(4);auto* pos=owner.take<int32_t>(24);auto* pages=owner.take<int32_t>(capacity/4);auto* pg=owner.host<int32_t>(size_t(capacity/4));for(int i=0;i<capacity/4;++i)pg[i]=i;q.memcpy(pages,pg,size_t(capacity/4)*4);host_step_keep_=host_step;host_positions_keep_=host_positions;step_keep_=step;positions_keep_=pos;pages_keep_=pages;state.reset(new DraftQsaTransaction(committed,working,capacity,host_step,host_positions,step,pos,pages));
    incoming=fl(10240);committed_h=fl(10240);q.memset(committed_h,0,10240*4);token=owner.take<int32_t>(1);host_token=owner.host<int32_t>(1);draft.reset(new OriginalDraftStep53(*state,committed_h,S.next_hidden,&q));q.wait_and_throw();
 }
 ~OriginalDraftSession53(){const strata::core::OnDevice on(1);owner.retire();}
 OriginalDraftSession53(const OriginalDraftSession53&)=delete;
 OriginalDraftSession53& operator=(const OriginalDraftSession53&)=delete;
 int64_t cells(OriginalDraftSlotOwner53 who)const {check(who);return state->committed_cells();}
 void advance_actual(OriginalDraftSlotOwner53 who,int32_t accepted_next,const float* actual_last_hidden){
  const strata::core::OnDevice on(1);check(who);
  try {
   if(active_ || !actual_last_hidden || accepted_next<0 || accepted_next>=248320 ||
      sycl::get_pointer_type(actual_last_hidden,q.get_context())!=sycl::usm::alloc::device || sycl::get_pointer_device(actual_last_hidden,q.get_context())!=q.get_device())throw std::runtime_error("original private actual LAST residual/token required");
   // Copy the authoritative row into PRIVATE persistent staging before capture use.
   // Caller must have completed the matching target slot commit, not a guessed row.
   active_=true;has_logits_=false;*host_token=accepted_next;
   q.memcpy(incoming,actual_last_hidden,10240*4);q.memcpy(token,host_token,4);
   draft->prepare_candidate(accepted_next);draft->record(token,incoming,weights_keep_->weights(),S);q.wait_and_throw();
   draft->settle_fixture(true);q.wait_and_throw();active_=false;has_logits_=true;
  }catch(...){failed=true;throw;}
 }
 int32_t propose_actual(OriginalDraftSlotOwner53 who){
  const strata::core::OnDevice on(1);check(who);
  try{if(active_ || !has_logits_)throw std::runtime_error("original private logits not committed/current");
   std::vector<float> logits(248320);q.memcpy(logits.data(),S.logits,logits.size()*4).wait_and_throw();int32_t best=0;
   for(int32_t i=0;i<248320;++i){if(!std::isfinite(logits[i]))throw std::runtime_error("original own head nonfinite");if(logits[i]>logits[best])best=i;}return best;
  }catch(...){failed=true;throw;}
 }
 void drain(OriginalDraftSlotOwner53 who){const strata::core::OnDevice on(1);check(who);try{q.wait_and_throw();}catch(...){failed=true;throw;}}
 // Reuse already-reserved private buffers only AFTER caller reset the actual
 // target slot to an empty prefix. No shared weight upload or GPU allocation.
 void reset_fresh(OriginalDraftSlotOwner53 old_owner,OriginalDraftSlotOwner53 next){
  const strata::core::OnDevice on(1);check(old_owner);
  try{
   if(active_ || !next.rid || next.generation<=slot_.generation)throw std::runtime_error("fresh slot requires drained transaction/new generation");q.wait_and_throw();
   auto zero=[&](const DraftBank& b){q.memset(b.key,0,size_t(capacity_)*512*2);q.memset(b.value,0,size_t(capacity_)*512*2);q.memset(b.tail,0,384*4);q.memset(b.dead,0,128*4);q.memset(b.pooled,0,size_t(capacity_/4+2)*128*4);q.memset(b.block_pos,0,4);};
   zero(committed_bank_);zero(working_bank_);q.memset(committed_h,0,10240*4);q.wait_and_throw();
   draft.reset();state.reset(new DraftQsaTransaction(committed_bank_,working_bank_,capacity_,host_step_keep_,host_positions_keep_,step_keep_,positions_keep_,pages_keep_));
   draft.reset(new OriginalDraftStep53(*state,committed_h,S.next_hidden,&q));has_logits_=false;slot_=next;
  }catch(...){failed=true;throw;}
 }
 // Bounded diagnostic ONLY, no serialization/import/current serving authority.
 OriginalDraftPrivateSnapshot53 diagnostic_snapshot(OriginalDraftSlotOwner53 who){
  const strata::core::OnDevice on(1);check(who);
  try{
   if(active_ || !has_logits_)throw std::runtime_error("private diagnostic requires committed/current head");q.wait_and_throw();
   OriginalDraftPrivateSnapshot53 out;out.cells=state->committed_cells();
   auto append=[&](std::vector<uint8_t>& bytes,const void* src,size_t n){size_t at=bytes.size();bytes.resize(at+n);q.memcpy(bytes.data()+at,src,n).wait_and_throw();};
   auto bank=[&](const DraftBank& b,std::vector<uint8_t>& bytes){append(bytes,b.key,size_t(capacity_)*512*2);append(bytes,b.value,size_t(capacity_)*512*2);append(bytes,b.tail,384*4);append(bytes,b.dead,128*4);append(bytes,b.pooled,size_t(capacity_/4+2)*128*4);append(bytes,b.block_pos,4);};
   bank(committed_bank_,out.committed);bank(working_bank_,out.working);append(out.hidden,committed_h,10240*4);
   out.logits.resize(248320);q.memcpy(out.logits.data(),S.logits,248320*4).wait_and_throw();
   for(void* p:owner.buffers)out.private_allocations.push_back(reinterpret_cast<uintptr_t>(p));return out;
  }catch(...){failed=true;throw;}
 }
 // Full original recurrence is PRIVATE: both K/V banks, tails/dead/pooled/blockpos,
 // committed hidden and counts. No legacy KV-only export/import API is exposed.
 // Reset/rewind/slot handoff/persisted restore remain explicitly unsupported.
};
}
