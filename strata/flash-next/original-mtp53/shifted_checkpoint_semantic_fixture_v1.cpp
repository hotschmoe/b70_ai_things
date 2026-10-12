#include "strata/kernels/original_mtp53_shared_sessions.hpp"
using namespace strata::kernels::original_mtp53;
std::unique_ptr<OriginalDraftSession53> compile_unbound53(std::shared_ptr<const OriginalDraftWeights53> pack,const sycl::queue& last){auto s=std::make_unique<OriginalDraftSession53>(pack,OriginalDraftSlotOwner53{0,0},last,1);s->reset_fresh({0,0},{9,1});return s;}
void compile_shifted53(OriginalDraftSession53& s,OriginalDraftSlotOwner53 owner,const std::vector<int32_t>& inputs){auto c=s.export_shifted(owner,inputs,std::string(64,'a'),std::string(64,'b'));s.import_shifted(owner,c,c.envelope_sha,inputs,8678,std::string(64,'a'),std::string(64,'b'));s.release_owner(owner);s.reset_fresh({0,0},{10,1});}
