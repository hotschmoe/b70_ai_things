// ROOT CPU-only fixture. Synthetic generation macro must be supplied explicitly.
#include "native_allocation_attribution_producer_v2.hpp"
#include <algorithm>
#include <cassert>
#include <type_traits>
template<class T>using host_vector=strata::core::allocation_attribution::checkpoint_vector<T>;
int main(){
 if(STRATA_ALLOCATION_ATTRIBUTION_SOURCE_GENERATION<41)return 3;
 static_assert(sizeof(host_vector<unsigned char>)==sizeof(std::vector<unsigned char>));
 static_assert(alignof(host_vector<unsigned char>)==alignof(std::vector<unsigned char>));
 static_assert(std::allocator_traits<strata::core::allocation_attribution::HostCacheAllocator<unsigned char,strata::core::allocation_attribution::checkpoint_payload>>::is_always_equal::value);
 std::vector<unsigned char> ordinary(257);for(size_t i=0;i<ordinary.size();++i)ordinary[i]=static_cast<unsigned char>(i);
 {host_vector<unsigned char> payload(ordinary.begin(),ordinary.end());assert(std::equal(payload.begin(),payload.end(),ordinary.begin()));auto copied=payload;host_vector<unsigned char> moved=std::move(copied);assert(copied.empty());assert(moved==payload);payload.reserve(1024);payload.resize(513,19);assert(payload[512]==19);payload.resize(1);payload.shrink_to_fit();host_vector<host_vector<unsigned char>> directory;directory.push_back(std::move(moved));auto clone=directory;assert(clone==directory);}
 if(strata::core::allocation_attribution::enabled()){auto& t=strata::core::allocation_attribution::tracker();std::lock_guard<std::mutex> lock(t.mutex);assert(t.live.empty());}
 std::puts("ALLOC_HOST_CPU_RESULT original_bytes_preserved=1 all_scoped_owners_released=1 native_runtime_qualified=0");return 0;
}
