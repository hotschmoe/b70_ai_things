// ROOT CPU-only successful allocator/free tracking ordering regression.
#include "native_allocation_attribution_producer_v2.hpp"
#include <thread>
#include <cassert>
using namespace strata::core::allocation_attribution;
int main(){
 if(!enabled())return 3;std::allocator<unsigned char> api;auto* first=api.allocate(16);allocation_success(first,16,"host_heap","","host_process",-1,"host_vector_api",0,1,16);AllocationKey key{"","host_process",reinterpret_cast<uintptr_t>(first)};
 std::atomic<unsigned char*> next{nullptr};std::atomic<bool> attempting{false},registered{false};bool actual_reused=false;
 std::thread registrar([&]{unsigned char* p;while(!(p=next.load()))std::this_thread::yield();attempting=true;allocation_success(p,16,"host_heap","","host_process",-1,"host_vector_api",0,1,16);registered=true;});
 release_owned(key,[&]{api.deallocate(first,16);auto* second=api.allocate(16);actual_reused=second==first;next=second;while(!attempting.load())std::this_thread::yield();assert(!registered.load());});
 registrar.join();assert(registered.load());auto* second=next.load();release_owned(AllocationKey{"","host_process",reinterpret_cast<uintptr_t>(second)},[&]{api.deallocate(second,16);});
 {auto& t=tracker();std::lock_guard<std::mutex> lock(t.mutex);assert(t.live.empty());}
 std::printf("ALLOC_REUSE_CPU_RESULT registration_waited_for_successful_free_retirement=1 actual_pointer_reused=%d native_runtime_qualified=0\n",actual_reused?1:0);return 0;
}
