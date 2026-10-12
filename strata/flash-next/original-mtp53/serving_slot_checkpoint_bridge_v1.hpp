#pragma once
// Source-only original draft serving seam. No existing batch/MTP guard is removed.
#include "strata/kernels/original_mtp53_shared_sessions.hpp"
#include "strata/core/verify.hpp"
#include <array>
#include <vector>
namespace strata::kernels::original_mtp53 {
struct OriginalSlotCompletedRows53 {
 int slot,first_row,row_count;OriginalDraftSlotOwner53 owner;int64_t base_position;
 int32_t guess;std::array<int32_t,2> target_picks;
 int remaining;bool eos_first,stop_requested;
};
class OriginalServingSlots53 final {
 struct Slot {std::unique_ptr<OriginalDraftSession53> session;OriginalDraftSlotOwner53 owner;bool live=false,stop=false;};
 std::shared_ptr<const OriginalDraftWeights53> pack_;strata::core::Verifier* first_=nullptr;strata::core::Verifier* last_=nullptr;std::vector<Slot> slots_;bool failed_=false;
 Slot& get(int i,OriginalDraftSlotOwner53 who){if(failed_ || i<0 || size_t(i)>=slots_.size())throw std::runtime_error("original serving slot failure/index");auto& s=slots_[size_t(i)];if(who.rid!=s.owner.rid || who.generation!=s.owner.generation)throw std::runtime_error("original serving stale RID/generation");return s;}
public:
 // Construct/reserve all sessions BEFORE sizing either target expert cache.
 OriginalServingSlots53(std::shared_ptr<const OriginalDraftWeights53> pack,const sycl::queue& last_queue,int capacity_slots)
 :pack_(std::move(pack)){
  if(!pack_ || capacity_slots<1 || capacity_slots>6)throw std::runtime_error("original serving1..6 private slots required");
  pack_->require_last(last_queue,1);slots_.resize(size_t(capacity_slots));
  for(int i=0;i<capacity_slots;++i){auto& s=slots_[size_t(i)];s.owner={uint64_t(i+1),1};s.session=std::make_unique<OriginalDraftSession53>(pack_,s.owner,last_queue,1);}
 }
 // After Verifier initialization, bind actual pair once; private allocation was
 // already reserved before either expert cache was sized.
 void bind_target(strata::core::Verifier& first,strata::core::Verifier& last){
  if(failed_ || first_ || &first==&last || first.device()!=0 || last.device()!=1 || !last.stream())throw std::runtime_error("original serving exact paired target required");
  pack_->require_last(*last.stream(),1);first_=&first;last_=&last;
 }
 void begin_fresh(int slot,OriginalDraftSlotOwner53 next,int64_t actual_target_prefix){
  if(failed_ || slot<0 || size_t(slot)>=slots_.size())throw std::runtime_error("original serving admission failed");auto& s=slots_[size_t(slot)];
  if(!first_ || !last_ || s.live || actual_target_prefix!=0)throw std::runtime_error("original serving fresh admission requires BOTH target stages reset to empty");
  try{s.session->reset_fresh(s.owner,next);s.owner=next;s.stop=false;s.live=true;}catch(...){failed_=true;throw;}
 }
 // This call is fed from LAST actual prefill callback with the accepted NEXT
 // prompt token; original recurrence must catch up every teacher row, not just KV.
 void teacher_committed(int slot,OriginalDraftSlotOwner53 who,int64_t position,int32_t next_token,const float* actual_last_hidden){
  auto& s=get(slot,who);try{if(!s.live || s.stop || s.session->cells(who)!=position)throw std::runtime_error("original serving teacher prefix differs");s.session->advance_actual(who,next_token,actual_last_hidden);}catch(...){failed_=true;throw;}
 }
 int32_t proposal(int slot,OriginalDraftSlotOwner53 who){auto& s=get(slot,who);if(!s.live || s.stop)throw std::runtime_error("original serving inactive/cancelled proposal");return s.session->propose_actual(who);}
 // Invoked instead of commit_slot_prefixes followed by stock draft(). No global
 // per-slot main.run: the caller has ALREADY run the real packed target window.
 // Commit waits all actual stages in SDK58, then copy/catchup BEFORE BT/BDONE or
 // any next verifier window can overwrite LAST residual rows. <=6 slots/8 rows (up to4 full T2 groups).
 // Real tail T1 contains one pick; unused second pick is not consumed.
 void commit_and_catchup(const std::vector<OriginalSlotCompletedRows53>& rows,std::vector<int>& keep,std::string& error){
  if(failed_ || !first_ || !last_ || rows.empty() || rows.size()>slots_.size() || keep.size()!=slots_.size())throw std::runtime_error("original serving bounded window differs");
  try{
   std::fill(keep.begin(),keep.end(),0);std::vector<bool> seen(slots_.size(),false);int end=0;
   for(const auto& r:rows){auto& s=get(r.slot,r.owner);if(!s.live || seen[size_t(r.slot)] || r.first_row!=end || (r.row_count!=1 && r.row_count!=2) || end+r.row_count>8 || r.base_position<0 || r.base_position+r.row_count>4096 || r.remaining<1 || s.session->cells(r.owner)!=r.base_position)throw std::runtime_error("original serving packed owner/position/group differs");seen[size_t(r.slot)]=true;end+=r.row_count;
    for(int j=0;j<r.row_count;++j)if(r.target_picks[size_t(j)]<0 || r.target_picks[size_t(j)]>=248320)throw std::runtime_error("original serving target pick out of range");
    const bool actual_eos=r.target_picks[0]==248044 || r.target_picks[0]==248046;
    if(r.eos_first!=actual_eos)throw std::runtime_error("original serving actual EOS differs");
    keep[size_t(r.slot)]=(r.row_count==2 && r.target_picks[0]==r.guess && !actual_eos && !r.stop_requested && !s.stop && r.remaining>=2 && r.base_position+3<=4096)?2:1;
   }
   if(!first_->commit_slot_prefixes(keep.data(),error))throw std::runtime_error(error);
   // SDK58 commit_slot_prefixes waits cs_ then recurses to next stage.
   if(!last_->stream() || last_->stream()->get_context()!=pack_->queue().get_context() || last_->stream()->get_device()!=pack_->queue().get_device())throw std::runtime_error("original serving LAST context changed");const float* residual=first_->final_R_all();if(!residual)throw std::runtime_error("actual LAST residual absent");
   for(const auto& r:rows){auto& s=get(r.slot,r.owner);for(int j=0;j<keep[size_t(r.slot)];++j)s.session->advance_actual(r.owner,r.target_picks[size_t(j)],residual+size_t(r.first_row+j)*10240);
    if(s.session->cells(r.owner)!=r.base_position+keep[size_t(r.slot)])throw std::runtime_error("original serving completed prefix catchup differs");
   }
  }catch(...){failed_=true;throw;}
 }
 bool request_cancel(int slot,OriginalDraftSlotOwner53 who){auto& s=get(slot,who);if(!s.live)return false;s.stop=true;return true;}
 // Terminal/cancel must follow catchup for ALL actual committed rows including
 // EOS/replacement. Retain state for diagnostics; no speculative next proposal.
 void finish(int slot,OriginalDraftSlotOwner53 who,int64_t actual_target_prefix){auto& s=get(slot,who);try{if(!s.live || s.session->cells(who)!=actual_target_prefix)throw std::runtime_error("original serving terminal prefix differs");s.session->drain(who);s.live=false;}catch(...){failed_=true;throw;}}
 // Complete original payload; caller pairs it with BOTH real target snapshots
 // and pins payload_sha in owned cache metadata. No KV-only publication.
 OriginalCheckpoint53 checkpoint(int slot,OriginalDraftSlotOwner53 who,const std::vector<int32_t>& actual_target_prefix,const std::string& draft_contract,const std::string& target_contract){
  auto& s=get(slot,who);return s.session->export_checkpoint(who,actual_target_prefix,draft_contract,target_contract);
 }
 // Caller has independently restored and byte-verified BOTH real target stages
 // at actual_target_prefix/current_target_contract. A new generation is required;
 // source owner may differ for a legitimate shared-prefix cache hit.
 void restore_checkpoint(int slot,OriginalDraftSlotOwner53 next,const OriginalCheckpoint53& c,const std::string& pinned_digest,const std::vector<int32_t>& actual_target_prefix,const std::string& current_draft_contract,const std::string& current_target_contract){
  if(failed_ || !first_ || !last_ || slot<0 || size_t(slot)>=slots_.size())throw std::runtime_error("original cache destination unavailable");auto& s=slots_[size_t(slot)];
  validate_checkpoint53(c);
  if(s.live || !next.rid || next.generation<=s.owner.generation || c.payload_sha!=pinned_digest || c.payload_sha!=checkpoint_digest53(c) || c.prefix!=actual_target_prefix || c.draft_contract_sha!=current_draft_contract || c.target_contract_sha!=current_target_contract)throw std::runtime_error("original complete cache prefix/source/pin differs");
  try{s.session->reset_fresh(s.owner,next);s.owner=next;s.session->import_checkpoint(next,c,pinned_digest,actual_target_prefix,current_draft_contract,current_target_contract);s.stop=false;s.live=true;}catch(...){failed_=true;throw;}
 }
 void refuse_legacy_cache_restore()const{throw std::runtime_error("original serving cache restore requires full original banks/indexer/hidden/logits/counts and BOTH target prefix checkpoints; legacy KV-only unsupported");}
};
}
