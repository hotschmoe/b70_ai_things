#pragma once
#include "strata/kernels/original_mtp53_joint_checkpoint.hpp"
#include "strata/kernels/original_mtp53_serving_slots.hpp"
#include "strata/core/on_device.hpp"
#include <cstdio>
namespace strata::kernels::original_mtp53 {
struct JointTargetSlot53 {strata::core::SessionState* state;strata::core::Verifier* verifier;int device,lb,le;};
inline std::shared_ptr<const OriginalJointCheckpoint53> capture_joint53(OriginalServingSlots53& draft,int slot,OriginalDraftSlotOwner53 owner,const std::array<JointTargetSlot53,2>& target,const std::array<strata::core::ConversationView,2>& views,const strata::core::ModelGeometry& g,const std::string& draft_contract,const std::string& target_contract,const std::function<OriginalDraftSlotOwner53()>& current_owner,std::vector<std::shared_ptr<const OriginalJointCheckpoint53>> branches={}){
 // Caller serializes slot admission/commit/branch mutation for this whole boundary.
 auto same=[&]{auto who=current_owner();return who.rid==owner.rid && who.generation==owner.generation;};if(!same() || views[0].ids!=views[1].ids)throw std::runtime_error("joint frozen slot/prefix differs");
 std::array<strata::core::SavedConversation,2> images;std::array<std::array<int32_t,2>,2> previous{};std::string error;
 for(int i=0;i<2;++i){const auto& t=target[i];if(!t.state || !t.verifier || t.device!=i || t.lb!=i*24 || t.le!=(i+1)*24 || t.state->layer_lo!=t.lb || t.state->layer_hi!=t.le || t.verifier->device()!=i || !t.verifier->stream())throw std::runtime_error("joint actual paired target slot carve differs");const strata::core::OnDevice on(i);dpct::get_current_device().queues_wait_and_throw();previous[i]={t.state->ple_prev[0],t.state->ple_prev[1]};if(!strata::core::conversation_snapshot_save(images[i],views[i],*t.state,g,(const strata::core::QsaState*)nullptr,error))throw std::runtime_error(error);}
 auto original=draft.checkpoint(slot,owner,views[0].ids,draft_contract,target_contract);if(!same())throw std::runtime_error("joint owner changed before atomic publication");return std::make_shared<OriginalJointCheckpoint53>(std::move(images),std::move(original),previous,std::move(branches));
}
inline void restore_joint53(const std::shared_ptr<const OriginalJointCheckpoint53>& image,const std::string& pinned_joint,OriginalServingSlots53& draft,int slot,OriginalDraftSlotOwner53 next,const std::array<JointTargetSlot53,2>& target,const strata::core::ModelGeometry& g,const std::string& current_draft_contract,const std::string& current_target_contract,const std::function<OriginalDraftSlotOwner53()>& current_owner){
 if(!image || image->digest()!=pinned_joint || image->original().draft_contract_sha!=current_draft_contract || image->original().target_contract_sha!=current_target_contract)throw std::runtime_error("joint current source/model/pin differs");auto who=current_owner();if(who.rid!=next.rid || who.generation!=next.generation)throw std::runtime_error("joint destination request identity differs");std::string error;
 draft.validate_checkpoint_restore(slot,next,image->original(),image->original().payload_sha,image->original().prefix,current_draft_contract,current_target_contract);
 for(int i=0;i<2;++i){const auto& t=target[i];if(!t.state || !t.verifier || t.device!=i || t.lb!=i*24 || t.le!=(i+1)*24 || t.state->layer_lo!=t.lb || t.state->layer_hi!=t.le || t.verifier->device()!=i || !t.verifier->stream())throw std::runtime_error("joint destination actual carve differs");if(!strata::core::conversation_snapshot_validate(image->targets()[i],*t.state,g,(const strata::core::QsaState*)nullptr,error))throw std::runtime_error(error);}
 // No invalid-image fallback after FIRST destination write. Fatal86 preserves
 // pending queue/resources; caller's owned actor retires/health-checks process.
 try{
  for(int i=0;i<2;++i){const strata::core::OnDevice on(i);dpct::get_current_device().queues_wait_and_throw();if(strata::core::conversation_snapshot_restore(image->targets()[i],*target[i].state,g,(const strata::core::QsaState*)nullptr,error)!=strata::core::ConversationRestore::restored)throw std::runtime_error(error);dpct::get_current_device().queues_wait_and_throw();target[i].state->ple_prev[0]=image->ple_previous()[i][0];target[i].state->ple_prev[1]=image->ple_previous()[i][1];
   strata::core::ConversationCheckpoint actual;actual.ids=image->original().prefix;if(!strata::core::conversation_checkpoint_save(actual,*target[i].state,g,error))throw std::runtime_error(error);const auto& expected=image->targets()[i].live;if(actual.gdn!=expected.gdn || actual.ple!=expected.ple || actual.tails!=expected.tails || actual.dead!=expected.dead || actual.block_pos!=expected.block_pos)throw std::runtime_error("joint target recurrent byte readback differs");if(image->targets()[i].kv.size()!=size_t(target[i].state->qsa_alloc))throw std::runtime_error("joint no legacy draft KV permitted");for(size_t j=0;j<image->targets()[i].kv.size();++j){uint64_t fp=0;if(!strata::core::conversation_kv_verify(image->targets()[i].kv[j],target[i].state->qsa_states[size_t(target[i].state->qsa_ord0)+j],g,image->original().cells,true,fp,error))throw std::runtime_error(error);}
  }
  draft.restore_checkpoint(slot,next,image->original(),image->original().payload_sha,image->original().prefix,current_draft_contract,current_target_contract);who=current_owner();if(who.rid!=next.rid || who.generation!=next.generation)throw std::runtime_error("joint owner changed before activation");
 }catch(...){std::fprintf(stderr,"MTP53_JOINT_FATAL partial_target_or_draft_restore=1 activation_refused=1\n");std::fflush(stderr);std::_Exit(86);}
 // Caller now installs matching BSlot ids/position and marks active. No earlier
 // cache/cancel/HTTP success publication is authorized by this function.
}
}
