// Compile-only real source58 Verifier and V4 private-session signatures.
#include "strata/kernels/original_mtp53_serving_slots.hpp"
#include <type_traits>
using namespace strata::kernels::original_mtp53;
static_assert(std::is_constructible_v<OriginalServingSlots53,std::shared_ptr<const OriginalDraftWeights53>,const sycl::queue&,int>);
void compile_bridge53(std::shared_ptr<const OriginalDraftWeights53> pack,const sycl::queue& queue,strata::core::Verifier& first,strata::core::Verifier& last){
 OriginalServingSlots53 slots(pack,queue,6);slots.bind_target(first,last);
 OriginalDraftSlotOwner53 who{123,2};slots.begin_fresh(0,who,0);
 slots.teacher_committed(0,who,0,248045,first.final_R_all());(void)slots.proposal(0,who);
 std::vector<int> keep(6);std::string error;
 // Actual one-row tail, second pick deliberately unused/-1.
 slots.commit_and_catchup({{0,0,1,who,1,-1,{198,-1},1,false,false}},keep,error);
 slots.request_cancel(0,who);slots.finish(0,who,2);
}
