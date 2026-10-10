// Draft default-OFF successful API accounting; no physical residency inference.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>
#ifndef STRATA_ALLOCATION_ATTRIBUTION_SOURCE_GENERATION
#define STRATA_ALLOCATION_ATTRIBUTION_SOURCE_GENERATION 0
#endif
namespace strata::core::allocation_attribution {
inline bool enabled(){static const bool on=[](){const char* p=std::getenv("STRATA_ALLOCATION_ATTRIBUTION");if(!p||!*p||!std::strcmp(p,"0"))return false;if(std::strcmp(p,"1")||STRATA_ALLOCATION_ATTRIBUTION_SOURCE_GENERATION<41)throw std::invalid_argument("allocation attribution requires1 and future compile-bound generation");return true;}();return on;}
inline uint64_t start_ticks(){std::ifstream f("/proc/self/stat");std::string line;std::getline(f,line);auto end=line.rfind(')');if(!f||end==std::string::npos)throw std::runtime_error("actual process start identity unavailable");std::istringstream in(line.substr(end+2));std::string value;for(int i=0;i<20;++i)if(!(in>>value))throw std::runtime_error("process start identity parse");return std::stoull(value);}
struct Context {uint64_t main_rid=0,slot_rid=0;const char* phase="startup";bool slot_observed=false;};
inline std::atomic<uint64_t>& main_rid(){static std::atomic<uint64_t> v{0};return v;}
inline Context& current(){static thread_local Context value;return value;}
struct Scope {Context previous;Scope(uint64_t main,uint64_t slot,const char* phase):previous(current()){current()={main,slot,phase,true};}~Scope(){current()=previous;}};
inline void main_command(uint64_t rid){if(enabled())main_rid().store(rid);}
inline Context context(){auto c=current();if(c.main_rid==0&&main_rid().load()!=0){c.main_rid=main_rid().load();c.slot_rid=0;c.phase="main_body_slot_unobserved";c.slot_observed=false;}return c;}
struct Entry {uint64_t id=0,bytes=0,owner_generation=0,element_size=0,element_count=0;int role_index=0;std::string device_uuid,queue_context,space,role;int stage=-1;std::map<std::pair<int,int>,std::pair<uint64_t,uint64_t>> experts;};
using AllocationKey=std::tuple<std::string,std::string,uintptr_t>;
struct Tracker {std::mutex mutex;std::map<AllocationKey,Entry> live;std::map<std::string,uint64_t> sequence;uint64_t next=0,next_owner=0;};
inline Tracker& tracker(){static Tracker t;return t;}
inline void safe_tag(const std::string& s){if(s.empty())throw std::invalid_argument("empty attribution tag");for(unsigned char c:s)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'||c==':'))throw std::invalid_argument("unsafe attribution tag");}
inline void emit(Tracker& t,const char* kind,const Entry& a,int layer=-1,int expert=-1,uint64_t offset=0,uint64_t bytes=0,const char* boundary="none"){
 safe_tag(a.space);safe_tag(a.role);safe_tag(a.queue_context);safe_tag(boundary);auto c=context();safe_tag(c.phase);auto& seq=t.sequence[a.device_uuid];if(seq>=65536)throw std::runtime_error("allocation attribution event quota");auto now=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();static const uint64_t process_start=start_ticks();const auto pid=long(getpid());
 std::ostringstream out;out<<std::fixed<<std::setprecision(9)<<"ALLOC_ATTR {\"kind\":\""<<kind<<"\",\"owner\":{\"engine_pid\":"<<pid<<",\"process_start_ticks\":"<<process_start<<",\"device_uuid\":";
 if(a.device_uuid.empty())out<<"null";else{safe_tag(a.device_uuid);out<<'"'<<a.device_uuid<<'"';}
 out<<",\"source_generation\":"<<STRATA_ALLOCATION_ATTRIBUTION_SOURCE_GENERATION<<"},\"sequence\":"<<seq+1<<",\"host_epoch\":"<<now<<",\"allocation_id\":\""<<pid<<'-'<<process_start<<'-'<<a.id<<"\",\"stage\":"<<a.stage<<",\"owner_generation\":"<<a.owner_generation<<",\"role_index\":"<<a.role_index<<",\"element_size\":"<<a.element_size<<",\"element_count\":"<<a.element_count<<",\"queue_context\":\""<<a.queue_context<<"\",\"role\":\""<<a.role<<"\",\"space\":\""<<a.space<<"\",\"api_result\":\"success\",\"bytes\":"<<(bytes?bytes:a.bytes)<<",\"offset\":"<<offset<<",\"layer\":"<<layer<<",\"expert\":"<<expert<<",\"copy_completion_observed\":"<<(!std::strcmp(kind,"expert_bytes_ready")?"true":"false")<<",\"existing_completion_boundary\":"<<(!std::strcmp(kind,"expert_bytes_ready")?"true":"false")<<",\"completion_boundary\":\""<<boundary<<"\",\"command_context\":{\"current_main_body_rid\":"<<c.main_rid<<",\"actual_slot_rid\":"<<c.slot_rid<<",\"phase\":\""<<c.phase<<"\"},\"actual_slot_context_observed\":"<<(c.slot_observed?"true":"false")<<",\"physical_residency_observed\":false}\n";
 const auto line=out.str();if(line.size()>16384)throw std::runtime_error("allocation event line quota");flockfile(stderr);const auto wrote=std::fwrite(line.data(),1,line.size(),stderr);const int flushed=std::fflush(stderr);funlockfile(stderr);if(wrote!=line.size()||flushed!=0)throw std::runtime_error("allocation attribution record write failed");++seq;
}

inline void allocation_success(void* p,uint64_t n,const char* space,const std::string& uuid,const std::string& queue,int stage,const char* role,int role_index=0,uint64_t element_size=0,uint64_t element_count=0){if(!enabled())return;if(!p||!n||((uuid.empty())!=(std::strcmp(space,"host_heap")==0))||(uuid.empty()?stage!=-1:stage<0))throw std::invalid_argument("actual successful allocation domain/stage required");auto& t=tracker();std::lock_guard<std::mutex> lock(t.mutex);auto key=AllocationKey{uuid,queue,reinterpret_cast<uintptr_t>(p)};if(t.live.count(key))throw std::invalid_argument("allocation pointer already live");Entry a;a.id=++t.next;a.bytes=n;a.owner_generation=++t.next_owner;a.element_size=element_size;a.element_count=element_count;a.role_index=role_index;a.device_uuid=uuid;a.queue_context=queue;a.space=space;a.role=role;a.stage=stage;t.live.emplace(key,a);emit(t,"allocation_success",a);}
inline void expert_ready(const void* p,uint64_t bytes,int layer,int expert,const char* boundary,const std::string& uuid,const std::string& queue){if(!enabled())return;if(!p||!bytes||layer<0||layer>=48||expert<0||expert>=512)throw std::invalid_argument("exact expert range required");auto& t=tracker();std::lock_guard<std::mutex> lock(t.mutex);auto address=reinterpret_cast<uintptr_t>(p);auto it=t.live.upper_bound(AllocationKey{uuid,queue,address});if(it==t.live.begin())throw std::invalid_argument("expert allocation missing");--it;if(std::get<0>(it->first)!=uuid||std::get<1>(it->first)!=queue)throw std::invalid_argument("expert actual owning context missing");auto& a=it->second;auto offset=address-std::get<2>(it->first);if(a.device_uuid.empty()||offset>a.bytes||bytes>a.bytes-offset)throw std::invalid_argument("expert outside direct backed USM");auto key=std::pair<int,int>{layer,expert};auto old=a.experts.find(key);if(old!=a.experts.end()){emit(t,"expert_bytes_released",a,layer,expert,old->second.first,old->second.second,"overwrite_after_existing_wait");a.experts.erase(old);}for(auto& pr:a.experts)if(!(offset+bytes<=pr.second.first||pr.second.first+pr.second.second<=offset))throw std::invalid_argument("overlapping expert ranges");a.experts[key]={offset,bytes};emit(t,"expert_bytes_ready",a,layer,expert,offset,bytes,boundary);}
template<class Free>inline void release_owned(const AllocationKey& pointer,Free free_api){
 if(!enabled()){free_api();return;}auto& t=tracker();std::lock_guard<std::mutex> lock(t.mutex);auto it=t.live.find(pointer);if(it==t.live.end())throw std::invalid_argument("foreign/double owned free");
 // Allocation APIs may return reused pointers in another thread, but their
 // registration cannot pass this lock until successful free retirement ends.
 free_api();auto& a=it->second;for(auto& pr:a.experts)emit(t,"expert_bytes_released",a,pr.first.first,pr.first.second,pr.second.first,pr.second.second,"owning_free_returned");a.experts.clear();emit(t,"allocation_release_success",a);t.live.erase(it);
}
// Only cache vectors use this allocator; observer bookkeeping remains excluded.
enum HostRole {generic_vector=0,checkpoint_payload=1,conversation_payload=2,conversation_directory=3};
inline const char* host_role(int role){return role==checkpoint_payload?"checkpoint_payload":role==conversation_payload?"conversation_payload":role==conversation_directory?"conversation_directory":"host_vector_api";}
template<class T,int Role=generic_vector>struct HostCacheAllocator:std::allocator<T>{using value_type=T;template<class U>struct rebind{using other=HostCacheAllocator<U,Role>;};HostCacheAllocator()=default;template<class U>HostCacheAllocator(const HostCacheAllocator<U,Role>&)noexcept{}T* allocate(size_t n){T* p=std::allocator<T>{}.allocate(n);try{if(n)allocation_success(p,uint64_t(n)*sizeof(T),"host_heap","","host_process",-1,host_role(Role),0,sizeof(T),n);}catch(...){std::allocator<T>{}.deallocate(p,n);throw;}return p;}void deallocate(T* p,size_t n)noexcept{auto key=AllocationKey{"","host_process",reinterpret_cast<uintptr_t>(p)};try{if(n)release_owned(key,[&]{std::allocator<T>{}.deallocate(p,n);});else std::allocator<T>{}.deallocate(p,n);}catch(const std::exception& e){std::fprintf(stderr,"ALLOC_ATTR_ERROR %s\n",e.what());std::fflush(stderr);std::_Exit(1);}}};
template<class T,class U,int Role>bool operator==(const HostCacheAllocator<T,Role>&,const HostCacheAllocator<U,Role>&){return true;}
template<class T,class U,int Role>bool operator!=(const HostCacheAllocator<T,Role>&,const HostCacheAllocator<U,Role>&){return false;}
template<class T>using host_vector=std::vector<T,HostCacheAllocator<T>>;
template<class T>using checkpoint_vector=std::vector<T,HostCacheAllocator<T,checkpoint_payload>>;
template<class T>using conversation_payload_vector=std::vector<T,HostCacheAllocator<T,conversation_payload>>;
template<class T>using conversation_directory_vector=std::vector<T,HostCacheAllocator<T,conversation_directory>>;
}
