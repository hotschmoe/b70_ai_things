// Optional host producer callbacks for SYCL numerical diagnostics; no GPU types.
#pragma once
#include <cstdint>
namespace strata::kernels::numerical_diag {
struct expert_view {
 const float* gate=nullptr;const float* up=nullptr;const void* hq=nullptr;
 const unsigned long long* group_ptr=nullptr;const std::int32_t* group_start=nullptr;
 const std::int32_t* groups=nullptr;const std::int32_t* entry_destination=nullptr;const std::int32_t* entry_token=nullptr;
 std::int64_t capacity=0,ffn=0;
};
struct hooks {
 void* user=nullptr;
 void (*shared_gu)(void*,const float*,const float*,int,int,void*)=nullptr;
 void (*shared_hq)(void*,const void*,int,int,void*)=nullptr;
 void (*expert_gu)(void*,const expert_view&,void*)=nullptr;
 void (*expert_hq)(void*,const expert_view&,void*)=nullptr;
};
struct entry_binding {
 const std::int32_t* router_ids=nullptr;const std::int32_t* residency=nullptr;
 const unsigned long long* mirror_table=nullptr;const unsigned long long* slot_offsets=nullptr;
 const std::uint8_t* cache_base=nullptr;std::int64_t slots=0,uniform_blob_bytes=0;
};
// One GPU work item per bounded diagnostic call: error / coverage updates are
// serialized on the source queue. Existing shared join protects final readout.
inline bool selected_entry(int token,int destination,int expert){return token==0&&destination>=0&&destination<10&&expert>=0&&expert<512;}
}
