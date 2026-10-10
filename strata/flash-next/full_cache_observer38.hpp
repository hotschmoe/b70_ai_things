// New source38 host-only diagnostics. Default OFF; no model/cache policy input.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include "strata/core/prefix_lifecycle.hpp"
namespace strata::core::full_cache38 {
struct config {
 bool enabled=false;std::array<uint64_t,6> selected{};unsigned count=0;
 config(){const char* flag=std::getenv("STRATA_FULL_CACHE_OBSERVER38");if(!flag||!*flag||!std::strcmp(flag,"0"))return;
  if(std::strcmp(flag,"1"))throw std::invalid_argument("fullcache38 enabled expects0|1");enabled=true;
  const char* p=std::getenv("STRATA_FULL_CACHE_CAPTURE_RIDS38");if(!p||!*p)throw std::invalid_argument("fullcache38 requires1..6 actual RIDs");
  while(*p){if(count==selected.size()||*p<'1'||*p>'9')throw std::invalid_argument("fullcache38 positive canonical RID list");errno=0;char* end=nullptr;const uint64_t rid=std::strtoull(p,&end,10);if(errno||!rid||end==p||(*end&&*end!=','))throw std::invalid_argument("fullcache38 RID parse/overflow");for(unsigned i=0;i<count;++i)if(selected[i]==rid)throw std::invalid_argument("fullcache38 duplicate RID");selected[count++]=rid;p=end;if(*p){++p;if(!*p)throw std::invalid_argument("fullcache38 trailing RID separator");}}
 }
};
inline const config& settings(){static const config c;return c;}
inline bool enabled(){return settings().enabled;}
inline bool selected(uint64_t rid){const auto& c=settings();if(!c.enabled)return true;for(unsigned i=0;i<c.count;++i)if(c.selected[i]==rid)return true;return false;}
struct owner {uint64_t rid=0,slotgen=0,command=0;int slot=-1;};
inline owner& current(){static owner value;return value;}
inline std::mutex& mutex(){static std::mutex value;return value;}
inline uint64_t next(){static std::atomic<uint64_t> value{0};return ++value;}
template<class Tokens>inline void begin(uint64_t rid,uint64_t slotgen,int slot,const Tokens& ids,bool fresh,bool pin_present,int64_t pin){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());auto& c=current();c={rid,slotgen,next(),slot};char digest[65];prefix_lifecycle::token_digest(ids,digest);
 std::fprintf(stderr,"FC38 {\"kind\":\"request_begin\",\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"slot\":%d,\"slotgen\":%llu,\"tokens\":%zu,\"input_sha256_le32\":\"%s\",\"fresh\":%s,\"pin_present\":%s,\"pin\":%lld,\"raw_requested\":%s,\"source_normal_dispatch\":\"%s\"}\n",long(getpid()),(unsigned long long)c.command,(unsigned long long)rid,slot,(unsigned long long)slotgen,ids.size(),digest,fresh?"true":"false",pin_present?"true":"false",(long long)pin,selected(rid)?"true":"false",slot<0?"GEN":"BGEN");
}
inline void work(int64_t reused,int64_t read_from,int64_t reread_to,int64_t reached,int64_t prompt,int64_t generated,bool cancelled,const char* finish){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto c=current();
 std::fprintf(stderr,"FC38 {\"kind\":\"request_work\",\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"slot\":%d,\"slotgen\":%llu,\"reused\":%lld,\"read_from\":%lld,\"reread_to\":%lld,\"prompt_reached\":%lld,\"prompt\":%lld,\"generated\":%lld,\"cancelled\":%s,\"finish\":\"%s\"}\n",long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,c.slot,(unsigned long long)c.slotgen,(long long)reused,(long long)read_from,(long long)reread_to,(long long)reached,(long long)prompt,(long long)generated,cancelled?"true":"false",finish);
}
template<class Chain>inline void points(const char* phase,const Chain& chain,int64_t cap,int64_t tail,int64_t victim=-1){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto c=current();size_t total=0;for(const auto& point:chain)total+=point.bytes();
 std::fprintf(stderr,"FC38 {\"kind\":\"checkpoint_inventory\",\"event\":%llu,\"phase\":\"%s\",\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"slot\":%d,\"slotgen\":%llu,\"cap\":%lld,\"actual_victim\":%lld,\"chain_capacity_bytes\":%zu,\"points\":[",(unsigned long long)next(),phase,long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,c.slot,(unsigned long long)c.slotgen,(long long)cap,(long long)victim,total);
 for(size_t i=0;i<chain.size();++i){const auto& p=chain[i];char digest[65];prefix_lifecycle::token_digest(p.ids,digest);std::fprintf(stderr,"%s{\"index\":%zu,\"tokens\":%zu,\"sha256_le32\":\"%s\",\"used\":%llu,\"tail\":%s,\"pinned\":%s,\"point_capacity_bytes\":%zu,\"stages\":%zu}",i?",":"",i,p.ids.size(),digest,(unsigned long long)p.used,(tail>=0&&int64_t(p.ids.size())==tail)?"true":"false",p.pinned?"true":"false",p.bytes(),p.stage_parts.size()+1);}
 std::fprintf(stderr,"],\"physical_allocator_reclamation_qualified\":false}\n");
}
// Independent metadata registry: does not change cache structures, their
// sizeof/capacity budgets, existing policy or the serial-only PCL active gate.
struct registry_capture {std::array<const void*,256> pointers{};std::array<uint64_t,256> ids{};size_t count=0;};
inline registry_capture& registry(){static registry_capture r;return r;}
inline uint64_t instance(const void* ptr){if(!enabled())return 0;auto& r=registry();for(size_t i=0;i<r.count;++i)if(r.pointers[i]==ptr)return r.ids[i];return 0;}
inline void forget(const void* ptr){if(!enabled())return;auto& r=registry();for(size_t i=0;i<r.count;++i)if(r.pointers[i]==ptr){r.pointers[i]=nullptr;r.ids[i]=0;}}
inline void restore(const void* ptr,uint64_t id){if(!enabled())return;auto& r=registry();for(size_t i=0;i<r.pointers.size();++i)if(!r.pointers[i]||r.pointers[i]==ptr){r.pointers[i]=ptr;r.ids[i]=id;r.count=std::max(r.count,i+1);return;}throw std::runtime_error("fullcache38 parked registry256 bound");}
inline void admit(const void* ptr){if(enabled())restore(ptr,next());}
template<class Entries>inline std::unique_ptr<registry_capture> capture(const Entries& entries){if(!enabled())return {};if(entries.size()>256)throw std::runtime_error("fullcache38 parked count256 bound");auto c=std::make_unique<registry_capture>();c->count=entries.size();for(size_t i=0;i<c->count;++i){c->pointers[i]=&entries[i];c->ids[i]=instance(c->pointers[i]);}return c;}
template<class Entries>inline void rebind(const Entries& entries,size_t removed,const std::unique_ptr<registry_capture>& before){if(!enabled())return;if(!before)throw std::runtime_error("fullcache38 missing actual erase registry");for(size_t i=0;i<before->count;++i)forget(before->pointers[i]);for(size_t i=0;i<entries.size();++i)restore(&entries[i],before->ids[i<removed?i:i+1]);}
template<class Entries>inline void parked(const char* phase,const Entries& entries,size_t budget,size_t bytes,size_t slots,int64_t victim=-1){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto c=current();std::fprintf(stderr,"FC38 {\"kind\":\"parked_inventory\",\"event\":%llu,\"phase\":\"%s\",\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"budget\":%zu,\"retained_bytes\":%zu,\"slots\":%zu,\"actual_victim\":%lld,\"entries\":[",(unsigned long long)next(),phase,long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,budget,bytes,slots,(long long)victim);
 for(size_t i=0;i<entries.size();++i){const auto& e=entries[i];char digest[65];prefix_lifecycle::token_digest(e.live.ids,digest);std::fprintf(stderr,"%s{\"index\":%zu,\"instance\":%llu,\"live_tokens\":%zu,\"live_sha256_le32\":\"%s\",\"snapshot_bytes\":%zu,\"pinned\":%s,\"checkpoint_count\":%zu,\"stages\":%zu}",i?",":"",i,(unsigned long long)instance(&e),e.live.ids.size(),digest,e.bytes(),e.pinned()?"true":"false",e.checkpoints.size(),e.stage_images.size()+1);}std::fprintf(stderr,"],\"physical_allocator_reclamation_qualified\":false}\n");
}
template<class Cache>inline void experts(const Cache& cache,int stage,int device,int64_t lb,int64_t le,int64_t n_expert){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto c=current();std::fprintf(stderr,"FC38 {\"kind\":\"expert_inventory\",\"event\":%llu,\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"stage\":%d,\"device\":%d,\"lb\":%lld,\"le\":%lld,\"slots\":%lld,\"resident\":%lld,\"logical_bytes\":%lld,\"full_logical_bytes\":%lld,\"map\":[",(unsigned long long)next(),long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,stage,device,(long long)lb,(long long)le,(long long)cache.slots(),(long long)cache.resident(),(long long)cache.bytes(),(long long)cache.full_bytes());bool comma=false;for(int64_t l=lb;l<le;++l)for(int64_t e=0;e<n_expert;++e){int slot=cache.slot_of(l,e);if(slot>=0){std::fprintf(stderr,"%s[%lld,%lld,%d]",comma?",":"",(long long)l,(long long)e,slot);comma=true;}}std::fprintf(stderr,"],\"physical_device_residency_qualified\":false}\n");
}
inline void host_memory(){if(!enabled())return;FILE* f=std::fopen("/proc/self/status","r");uint64_t rss_kib=0;bool found=false;char line[512];while(f&&std::fgets(line,sizeof line,f)){unsigned long long value=0;if(std::sscanf(line,"VmRSS: %llu kB",&value)==1){rss_kib=value;found=true;}}if(f)std::fclose(f);std::lock_guard<std::mutex> lock(mutex());const auto c=current();std::fprintf(stderr,"FC38 {\"kind\":\"host_memory\",\"pid\":%ld,\"command\":%llu,\"rid\":%llu,\"proc_VmRSS_observed\":%s,\"VmRSS_bytes\":%llu,\"device_physical_memory_qualified\":false}\n",long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,found?"true":"false",(unsigned long long)(rss_kib*1024));}
} // namespace strata::core::full_cache38
