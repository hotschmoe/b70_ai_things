#include "strata/kernels/original_mtp53_shared_sessions.hpp"
using namespace strata::kernels::original_mtp53;
OriginalCheckpoint53 compile_export53(OriginalDraftSession53& s,OriginalDraftSlotOwner53 who,const std::vector<int32_t>& actual_prefix,const std::string& draft,const std::string& target){return s.export_checkpoint(who,actual_prefix,draft,target);}
void compile_import53(OriginalDraftSession53& s,OriginalDraftSlotOwner53 who,const OriginalCheckpoint53& c,const std::string& expected_sha,const std::vector<int32_t>& actual_prefix,const std::string& draft,const std::string& target){s.import_checkpoint(who,c,expected_sha,actual_prefix,draft,target);}

#include "strata/kernels/original_mtp53_serving_slots.hpp"
void compile_slot_checkpoint53(OriginalServingSlots53& slots,int slot,OriginalDraftSlotOwner53 owner,OriginalDraftSlotOwner53 next,const std::vector<int32_t>& prefix,const std::string& draft,const std::string& target){auto c=slots.checkpoint(slot,owner,prefix,draft,target);slots.restore_checkpoint(slot,next,c,c.payload_sha,prefix,draft,target);}
