// Compile-only semantic fixture. No main, no runtime, no GGUF or device access.
#include "strata/kernels/original_mtp53_shared_sessions.hpp"
#include <type_traits>
using namespace strata::kernels::original_mtp53;
static_assert(std::is_constructible_v<OriginalDraftWeights53,const char*,const OriginalDraftIdentity53&,const sycl::queue&,int>);
static_assert(std::is_constructible_v<OriginalDraftSession53,std::shared_ptr<const OriginalDraftWeights53>,OriginalDraftSlotOwner53,const sycl::queue&,int>);
static_assert(!std::is_copy_constructible_v<OriginalDraftWeights53>);
static_assert(!std::is_copy_constructible_v<OriginalDraftSession53>);
static_assert(std::is_same_v<decltype(std::declval<const OriginalDraftWeights53&>().weights()),const StepWeights53&>);
// Ordinary non-template definitions require real constructor/member bodies and
// complete destruction signatures to type-check under the production compiler.
std::shared_ptr<const OriginalDraftWeights53> compile_pack_signature53(const char* path,const OriginalDraftIdentity53& identity,const sycl::queue& last){
 return std::make_shared<OriginalDraftWeights53>(path,identity,last,1);
}
std::unique_ptr<OriginalDraftSession53> compile_session_signature53(std::shared_ptr<const OriginalDraftWeights53> pack,const sycl::queue& last,OriginalDraftSlotOwner53 slot){
 return std::make_unique<OriginalDraftSession53>(std::move(pack),slot,last,1);
}
int64_t compile_actual_slot_signature53(OriginalDraftSession53& session,OriginalDraftSlotOwner53 owner,int32_t token,const float* last_hidden){
 session.advance_actual(owner,token,last_hidden);session.drain(owner);(void)session.propose_actual(owner);return session.cells(owner);
}
