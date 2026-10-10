"""Deliberate unnumbered producer draft over exact corrected source40."""
import difflib,hashlib,json
from pathlib import Path
import prepare_native_layer3_qsa_observer_v2 as base
ROOT=base.ROOT;HERE=base.HERE
REFERENCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T122458Z-_0gr_xk8/source')
BASE_PLAN=base.PLAN;BASE_SHA=base.PLAN_SHA
H='include/strata/core/native_allocation_attribution.hpp'
S='sycl/include/strata/core/native_allocation_attribution_sycl.hpp'
PATCH=HERE/'patches/draft-default-off-native-allocation-expert-attribution-v1.patch'
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Exact producer hook anchor missing/ambiguous '+a[:80])
 return s.replace(a,b,1)
def reconstruct():
 base.build_plan();_,old=base.reconstruct();old=dict(old)
 for name in ('sycl/src/core/expert_cache.cpp','include/strata/core/expert_cache.hpp','sycl/src/core/conversation_snapshot_test.cpp','src/core/conversation_file_test.cpp'):old[name]=(REFERENCE/name).read_text()
 new=dict(old);new[H]=(HERE/'native_allocation_attribution_producer_v1.hpp').read_text();new[S]=(HERE/'native_allocation_attribution_sycl_v1.hpp').read_text()
 # Returned API hooks; existing API/free/queue waits retained verbatim.
 path='sycl/include/strata/core/stage_expert_mirror.hpp';s=new[path];s=once(s,'#include <sycl/sycl.hpp>','#include <sycl/sycl.hpp>\n#include "strata/core/native_allocation_attribution_sycl.hpp"')
 s=once(s,'        sycl::free(pointer,q);','        const auto allocation_key=allocation_attribution::usm_key(pointer,q);\n        sycl::free(pointer,q);\n        allocation_attribution::release_returned(allocation_key);');new[path]=s
 path='sycl/src/core/gguf_expert_source.cpp';s=new[path];s=once(s,'    if (mirror_ || lb<0', '    if(allocation_attribution::enabled()&&trace_stage<0)throw std::invalid_argument("actual mirror attribution stage missing");\n    if (mirror_ || lb<0')
 s=once(s,'            owner->segments.push_back({pointer,plan.sizes[i]});','            allocation_attribution::usm_success(pointer,plan.sizes[i],owner->q,trace_stage,"stage_mirror_segment");\n            owner->segments.push_back({pointer,plan.sizes[i]});')
 s=once(s,'        if(owner->host)owner->trace_register(owner->host,bytes,"contiguous_host",-1);','        if(owner->host)allocation_attribution::usm_success(owner->host,bytes,owner->q,trace_stage,"stage_mirror_host");\n        if(owner->host)owner->trace_register(owner->host,bytes,"contiguous_host",-1);')
 s=once(s,'    if (bad) { err="stage mirror: original GGUF gather failed"; return {}; }','    if (bad) { err="stage mirror: original GGUF gather failed"; return {}; }\n    for(size_t j=0;j<take.size();++j)allocation_attribution::usm_ready(destinations[j],lay.bytes[size_t(take[j].first)],int(take[j].first),int(take[j].second),owner->q,"host_source_workers_joined");')
 s=once(s,'    owner->trace_register(owner->table,table.size()*sizeof(table[0]),"device_table",-1);','    allocation_attribution::usm_success(owner->table,table.size()*sizeof(table[0]),owner->q,trace_stage,"stage_mirror_table");\n    owner->trace_register(owner->table,table.size()*sizeof(table[0]),"device_table",-1);');new[path]=s
 path='sycl/include/strata/core/slot_session_arena.hpp';s=new[path];s=once(s,'#include <sycl/sycl.hpp>','#include <sycl/sycl.hpp>\n#include "strata/core/native_allocation_attribution_sycl.hpp"');s=once(s,'            trace_attempt();base_=sycl::malloc_device(bytes,queue_);bytes_=bytes;', '            if(allocation_attribution::enabled()&&stage_<0)throw std::invalid_argument("slot attribution stage missing");\n            trace_attempt();base_=sycl::malloc_device(bytes,queue_);bytes_=bytes;');s=once(s,'            trace_register(base_,bytes_,"slot_arena");','            allocation_attribution::usm_success(base_,bytes_,queue_,stage_,"slot_session_arena");\n            trace_register(base_,bytes_,"slot_arena");');s=once(s,'        if(base_){sycl::free(base_,queue_);base_=nullptr;}','        if(base_){const auto allocation_key=allocation_attribution::usm_key(base_,queue_);sycl::free(base_,queue_);allocation_attribution::release_returned(allocation_key);base_=nullptr;}');new[path]=s
 path='include/strata/core/expert_cache.hpp';s=new[path];s=once(s,'private:\n','public:\n    void allocation_observer_stage(int stage){allocation_stage_=stage;}\nprivate:\n    int allocation_stage_=-1;\n    void allocation_observer_ready(int32_t slot,size_t bytes,const char* boundary);\n');new[path]=s
 path='sycl/src/core/expert_cache.cpp';s=new[path];s=once(s,'#include "strata/core/expert_cache.hpp"','#include "strata/core/expert_cache.hpp"\n#include "strata/core/native_allocation_attribution_sycl.hpp"')
 s=once(s,'    close();\n    if (n_slots <= 0)', '    close();\n    if(allocation_attribution::enabled()&&(allocation_stage_<0||seg_req_>0||g_cache_vmm))throw std::invalid_argument("allocation attribution requires known stage and fixed backed USM");\n    if (n_slots <= 0)')
 s=once(s,'    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer','    if(allocation_attribution::enabled()){if(!segs_.empty()||vmm_)throw std::invalid_argument("allocation attribution VA/VMM unsupported");allocation_attribution::usm_success(base_,want,dpct::get_in_order_queue(),allocation_stage_,"expert_cache_arena");}\n    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer')
 s=once(s,'        sycl::free(base_, dpct::get_in_order_queue());','        const auto allocation_key=allocation_attribution::usm_key(base_,dpct::get_in_order_queue());\n        sycl::free(base_, dpct::get_in_order_queue());\n        allocation_attribution::release_returned(allocation_key);')
 start=s.index('bool ExpertCache::fill_slot(');end=s.index('bool ExpertCache::fill_slot_blocking',start);block=s[start:end];block=once(block,'    const size_t n =','    if(allocation_attribution::enabled())throw std::invalid_argument("asynchronous expert readiness unobserved; use existing blocking route");\n    const size_t n =');s=s[:start]+block+s[end:]
 for func,boundary in [('fill_slot_blocking','blocking_copy_wait_returned'),('fill_slot_queued','SYCL_queued_copy_existing_wait_returned')]:
  start=s.index('bool ExpertCache::'+func+'(');end=s.index('\ncatch (sycl::exception',start);block=s[start:end];block=once(block,'    ++fills_;','    allocation_observer_ready(slot,n,"'+boundary+'");\n    ++fills_;');s=s[:start]+block+s[end:]
 anchor='bool ExpertCache::sync_queued(std::string &err) try {'
 ready='''void ExpertCache::allocation_observer_ready(int32_t slot,size_t bytes,const char* boundary){
 if(!allocation_attribution::enabled())return;
 size_t match=SIZE_MAX;for(size_t i=0;i<residency_.size();++i)if(residency_[i]==slot){if(match!=SIZE_MAX)throw std::invalid_argument("duplicate actual expert slot identity");match=i;}
 if(match==SIZE_MAX||n_expert_<=0)throw std::invalid_argument("actual expert slot identity missing");
 allocation_attribution::usm_ready(device_slot(slot),bytes,int(match/size_t(n_expert_)),int(match%size_t(n_expert_)),dpct::get_in_order_queue(),boundary);
}

'''
 s=once(s,anchor,ready+anchor);new[path]=s
 # std::allocator hooks for retained payloads and segment-directory bytes.
 path='include/strata/core/conversation_buffer.hpp';s=new[path];s=once(s,'#include <vector>','#include <vector>\n#include "strata/core/native_allocation_attribution.hpp"');s=once(s,'    using Segment = std::vector<uint8_t>;','    using Segment = allocation_attribution::host_vector<uint8_t>;');s=once(s,'    std::vector<Segment> segments_;','    allocation_attribution::host_vector<Segment> segments_;');new[path]=s
 path='include/strata/core/conversation_cache.hpp';s=new[path];s=once(s,'    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;','    allocation_attribution::host_vector<uint8_t> gdn, ple, tails, dead, block_pos;');new[path]=s
 path='src/core/conversation_file.cpp';s=new[path];s=once(s,'template<class T> void vec(const std::vector<T>& v)','template<class T,class A> void vec(const std::vector<T,A>& v)');s=once(s,'template<class T> bool vec(std::vector<T>& v','template<class T,class A> bool vec(std::vector<T,A>& v');new[path]=s
 path='sycl/src/core/conversation_snapshot_test.cpp';s=new[path];s=once(s,'    std::vector<uint8_t> spare(sizes.dead);','    strata::core::allocation_attribution::host_vector<uint8_t> spare(sizes.dead);');new[path]=s
 path='src/core/conversation_file_test.cpp';s=new[path];s=once(s,'std::vector<uint8_t> bytes_of(size_t n, uint8_t seed)', 'strata::core::allocation_attribution::host_vector<uint8_t> bytes_of(size_t n, uint8_t seed)');s=once(s,'    std::vector<uint8_t> v(n);','    strata::core::allocation_attribution::host_vector<uint8_t> v(n);');new[path]=s
 # Actual startup stage binding and current main-body RID are distinct from slot owner.
 path='sycl/src/program/generate.cpp';s=new[path];s=once(s,'        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL")','        xcache.allocation_observer_stage(0);\n        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL")');s=once(s,'    for (auto& stp : stages) {\n        GpuStage& st = *stp;\n        const auto& lay = strata::kernels::cpu::expert_layout();','    int allocation_stage=0;\n    for (auto& stp : stages) {\n        GpuStage& st = *stp;\n        st.cache.allocation_observer_stage(++allocation_stage);\n        const auto& lay = strata::kernels::cpu::expert_layout();')
 anchor='            strata::core::full_cache38::begin(req_rid,req_slot_generation,admit_slot,ids,req_fresh,req_pin_present,req_pin);';s=once(s,anchor,anchor+'\n            strata::core::allocation_attribution::main_command(req_rid);')
 anchor='            const auto& slot=bs.at(size_t(b));if(!batch_chain_enabled||!slot.cached||slot.active||slot.img||slot.ids.empty()||!conversations.enabled())return true;';s=once(s,anchor,anchor+'\n            strata::core::allocation_attribution::Scope allocation_scope(strata::core::allocation_attribution::main_rid().load(),slot.request_id,"park_slot");');new[path]=s
 return old,new

def patch_bytes():
 old,new=reconstruct();return ''.join(''.join(difflib.unified_diff(old.get(n,'').splitlines(True),new[n].splitlines(True),fromfile='a/'+n if n in old else '/dev/null',tofile='b/'+n))for n in sorted(new)if old.get(n)!=new[n]).encode('ascii')
