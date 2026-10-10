// Source39 logical accounting only. No allocator/device/math policy input.
#pragma once
#include <array>
#include <limits>
#include "strata/core/full_cache_observer38.hpp"
namespace strata::core::full_cache_memory39 {
inline void lifetime(const char* kind,uint64_t rid,uint64_t slotgen,int slot,uint64_t window,int64_t pos,int64_t consumed,int64_t generated,int token,const char* finish,int continuation=-1){
 if(!full_cache38::enabled())return;std::lock_guard<std::mutex> lock(full_cache38::mutex());
 std::fprintf(stderr,"FC39 {\"kind\":\"native_lifetime\",\"event\":%llu,\"pid\":%ld,\"phase\":\"%s\",\"rid\":%llu,\"slotgen\":%llu,\"slot\":%d,\"batch_window\":%llu,\"position\":%lld,\"consumed_tokens\":%lld,\"generated\":%lld,\"token\":%d,\"finish\":\"%s\",\"continuation\":%d,\"actual_HTTP_client_terminal_qualified\":false}\n",(unsigned long long)full_cache38::next(),long(getpid()),kind,(unsigned long long)rid,(unsigned long long)slotgen,slot,(unsigned long long)window,(long long)pos,(long long)consumed,(long long)generated,token,finish?finish:"",continuation);
}
inline size_t add(size_t a,size_t b){if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("fullcache39 logical byte overflow");return a+b;}
struct peak {const void* cache=nullptr;size_t allocated=0,reserved=0;};
inline peak& state(const void* cache){static std::array<peak,256> states{};for(auto& p:states)if(p.cache==cache)return p;for(auto& p:states)if(!p.cache){p.cache=cache;return p;}throw std::runtime_error("fullcache39 cache scope256 bound");}
template<class Reuses>inline size_t local_reuse_bytes(const Reuses& reuse){size_t bytes=0;for(const auto& p:reuse)bytes=add(bytes,p.bytes());return bytes;}
inline void chains(size_t main,size_t slots,size_t budget,size_t transfer){
 if(!full_cache38::enabled())return;std::lock_guard<std::mutex> lock(full_cache38::mutex());const auto c=full_cache38::current();
 std::fprintf(stderr,"FC39 {\"kind\":\"chain_resources\",\"event\":%llu,\"pid\":%ld,\"current_main_body_command\":%llu,\"current_main_body_rid\":%llu,\"active_main_chain_bytes\":%zu,\"batch_slot_chain_bytes\":%zu,\"total_chain_bytes\":%zu,\"configured_chain_budget_bytes\":%zu,\"configured_transfer_budget_bytes\":%zu,\"actual_transient_transfer_peak_qualified\":false,\"whole_process_peak_qualified\":false}\n",(unsigned long long)full_cache38::next(),long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,main,slots,add(main,slots),budget,transfer);
}
template<class Cache>inline void expert_backing(const Cache& cache,int stage,int device){
 if(!full_cache38::enabled())return;std::lock_guard<std::mutex> lock(full_cache38::mutex());const auto c=full_cache38::current();
 std::fprintf(stderr,"FC39 {\"kind\":\"expert_backing_metadata\",\"event\":%llu,\"pid\":%ld,\"current_main_body_command\":%llu,\"current_main_body_rid\":%llu,\"stage\":%d,\"device\":%d,\"mapped_arena_metadata_bytes\":%lld,\"live_arena_bytes\":%lld,\"full_arena_bytes\":%lld,\"full_slots\":%lld,\"segmented\":%s,\"segment_bytes\":%lld,\"actual_device_residency_qualified\":false}\n",(unsigned long long)full_cache38::next(),long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,stage,device,(long long)cache.mapped_bytes(),(long long)cache.bytes(),(long long)cache.full_bytes(),(long long)cache.full_slots(),cache.segmented()?"true":"false",(long long)cache.segment_bytes());
}
inline void record(const void* cache,const char* phase,size_t budget,size_t parked,size_t reusable,size_t held=0,size_t incoming=0,size_t local_reuse=0,size_t allocated_snapshot=0){
 if(!full_cache38::enabled())return;
 const auto owned_pending=add(local_reuse,allocated_snapshot);
 const auto cache_total=add(parked,reusable),base=add(cache_total,held);
 const auto allocated=add(base,owned_pending),reserved=add(base,std::max(incoming,owned_pending));
 std::lock_guard<std::mutex> lock(full_cache38::mutex());auto& p=state(cache);p.allocated=std::max(p.allocated,allocated);p.reserved=std::max(p.reserved,reserved);const auto c=full_cache38::current();
 std::fprintf(stderr,"FC39 {\"kind\":\"logical_memory\",\"event\":%llu,\"pid\":%ld,\"current_main_body_command\":%llu,\"current_main_body_rid\":%llu,\"phase\":\"%s\",\"cache_scope\":%zu,\"budget_bytes\":%zu,\"parked_bytes\":%zu,\"reusable_bytes\":%zu,\"cache_total_bytes\":%zu,\"held_snapshot_bytes\":%zu,\"incoming_estimate_bytes\":%zu,\"local_reuse_bytes\":%zu,\"allocated_snapshot_bytes\":%zu,\"observed_owned_snapshot_bytes\":%zu,\"logical_reservation_bytes\":%zu,\"observed_owned_snapshot_peak_bytes\":%zu,\"logical_reservation_peak_bytes\":%zu,\"reservation_within_budget\":%s,\"incoming_estimate_overlaps_owned_pending\":true,\"whole_process_peak_qualified\":false,\"physical_reclamation_qualified\":false,\"device_physical_memory_qualified\":false}\n",(unsigned long long)full_cache38::next(),long(getpid()),(unsigned long long)c.command,(unsigned long long)c.rid,phase,reinterpret_cast<size_t>(cache),budget,parked,reusable,cache_total,held,incoming,local_reuse,allocated_snapshot,allocated,reserved,p.allocated,p.reserved,reserved<=budget?"true":"false");
}
} // namespace strata::core::full_cache_memory39
