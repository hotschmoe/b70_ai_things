// SOURCE PROPOSAL ONLY: root must integrate/build/requalify a NEW SDK; no source35 edit.
// Host metadata only. No device copy, allocation, scheduling or cache-policy change.
#pragma once
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <strata/core/batch_prefix_chain.hpp>
#include <strata/core/prefix_lifecycle.hpp>
namespace strata::core::cache_point_trace_proposal {
inline bool enabled(){static const bool v=[](){const char* p=std::getenv("STRATA_CACHE_POINT_TRACE");if(!p||!*p||!std::strcmp(p,"0"))return false;if(std::strcmp(p,"1"))throw std::invalid_argument("STRATA_CACHE_POINT_TRACE=0|1");return true;}();return v;}
inline uint64_t next(){static std::atomic<uint64_t> n{0};return ++n;}
inline std::mutex& mutex(){static std::mutex m;return m;}
// The caller binds actor=main|slot and slot/RID/generation to the actual owner.
// Victim is the ACTUAL already-selected index before checks.erase, never a recomputed index.
// as_tail flags supplied by the caller use the exact same array passed to eviction_victim.
inline void inventory(const char* phase,const char* actor,int slot,uint64_t rid,uint64_t generation,
                      const std::vector<ConversationCheckpoint>& chain,int64_t cap,
                      const bool* tail=nullptr,int64_t actual_victim=-1){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto event=next();
 if(std::strcmp(phase,"before_eviction")&&std::strcmp(phase,"after_eviction")&&std::strcmp(phase,"lookup")&&std::strcmp(phase,"pin_saved")&&std::strcmp(phase,"clone_import"))throw std::invalid_argument("exact cache point phase");
 if(std::strcmp(actor,"main")&&std::strcmp(actor,"slot"))throw std::invalid_argument("exact cache chain actor");
 if(actual_victim>=int64_t(chain.size())||actual_victim<-1)throw std::invalid_argument("actual checkpoint victim index");
 std::fprintf(stderr,"CACHE_POINT {\"event\":%llu,\"phase\":\"%s\",\"actor\":\"%s\",\"slot\":%d,\"rid\":%llu,\"generation\":%llu,\"cap\":%lld,\"chain_capacity_bytes\":%zu,\"actual_victim\":%lld,\"points\":[",(unsigned long long)event,phase,actor,slot,(unsigned long long)rid,(unsigned long long)generation,(long long)cap,batch_prefix::chain_bytes(chain),(long long)actual_victim);
 for(size_t i=0;i<chain.size();++i){const auto& c=chain[i];char digest[65];prefix_lifecycle::token_digest(c.ids,digest);std::fprintf(stderr,"%s{\"index\":%zu,\"tokens\":%zu,\"sha256_le32\":\"%s\",\"used\":%llu,\"pinned\":%s,\"tail\":%s,\"point_capacity_bytes\":%zu,\"stages\":%zu}",i?",":"",i,c.ids.size(),digest,(unsigned long long)c.used,c.pinned?"true":"false",tail&&tail[i]?"true":"false",batch_prefix::point_bytes(c),c.stage_parts.size()+1);}
 std::fprintf(stderr,"]}\n");
}
// Proposed host-side resident MAP inventory, not a physical-memory measurement.
// Caller invokes only after accepted commits/queues settle, under its real stage context.
template<class Cache>inline void expert_inventory(const Cache& cache,int stage,int device,int64_t lb,int64_t le,int64_t n_expert){
 if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());const auto event=next();
 std::fprintf(stderr,"CACHE_EXPERT {\"event\":%llu,\"stage\":%d,\"device\":%d,\"lb\":%lld,\"le\":%lld,\"n_expert\":%lld,\"live_slots\":%lld,\"resident_entries\":%lld,\"live_capacity_bytes\":%lld,\"full_capacity_bytes\":%lld,\"map\":[",(unsigned long long)event,stage,device,(long long)lb,(long long)le,(long long)n_expert,(long long)cache.slots(),(long long)cache.resident(),(long long)cache.bytes(),(long long)cache.full_bytes());
 bool comma=false;for(int64_t l=lb;l<le;++l)for(int64_t e=0;e<n_expert;++e){const auto slot=cache.slot_of(l,e);if(slot<0)continue;std::fprintf(stderr,"%s[%lld,%lld,%d]",comma?",":"",(long long)l,(long long)e,int(slot));comma=true;}std::fprintf(stderr,"]}\n");
}
} // namespace strata::core::cache_point_trace_proposal
