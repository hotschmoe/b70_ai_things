"""New source39 memory accounting over immutable source38, CPU only."""
import difflib,hashlib,json
from pathlib import Path
import prepare_full_cache_observer38_v1 as source38
ROOT=source38.ROOT;HERE=source38.HERE
BASE=HERE/'full-cache-observer38-engine-build-plan-v1.json'
BASE_SHA='548d978e893ec6d9b73b33e282badc77edbf633c7f027e6ed553b361a1475f9d'
H='include/strata/core/full_cache_memory39.hpp'
HEADER=HERE/'full_cache_memory39.hpp'
PATCH=HERE/'patches/0039-default-off-full-cache-logical-memory-accounting.patch'
sha=source38.sha;once=source38.once
def reconstruct():
 if sha(BASE)!=BASE_SHA:raise ValueError('Immutable source38 plan changed')
 _,old=source38.reconstruct();plan=json.loads(BASE.read_bytes())
 for n,s in old.items():
  if hashlib.sha256(s.encode()).hexdigest()!=plan['expected_patched_source_sha256'][n]:raise ValueError('Source38 reconstruction changed '+n)
 new=dict(old);new[H]=HEADER.read_text()
 path='include/strata/core/conversation_cache.hpp';s=new[path]
 s=once(s,'#include "strata/core/full_cache_observer38.hpp"','#include "strata/core/full_cache_observer38.hpp"\n#include "strata/core/full_cache_memory39.hpp"')
 anchor='    bool enabled() const { return budget_ != 0 && slots_ != 0; }'
 s=once(s,anchor,anchor+'''
    void observe_memory39(const char* phase,size_t held=0,size_t incoming=0,size_t local_reuse=0,size_t snapshot=0) const {
        if(!full_cache38::enabled())return;
        full_cache_memory39::record(this,phase,budget_,bytes_,reuse_.bytes(),held,incoming,local_reuse,snapshot);
    }''')
 s=once(s,'        reuse_ = {};\n        ConversationKvReuse candidate','        observe_memory39("retain_before");\n        reuse_ = {};\n        ConversationKvReuse candidate')
 s=once(s,'        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);','        if(full_cache38::enabled())observe_memory39("retain_candidate",0,0,0,candidate.bytes());\n        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);\n        observe_memory39("retain_after");')
 s=once(s,'        if (reuse_.unchanged_tokens <= 0) reuse_ = {};','        if (reuse_.unchanged_tokens <= 0) reuse_ = {};\n        observe_memory39("limit_reuse_after");')
 s=once(s,'    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }','    ConversationKvReuse take_reuse() {\n        observe_memory39("take_reuse_before");\n        auto out=std::exchange(reuse_, {});\n        if(full_cache38::enabled())observe_memory39("take_reuse_after",0,0,out.bytes());\n        return out;\n    }')
 s=once(s,'        return out;\n    }\n\n    // Reserve before allocating','        if(full_cache38::enabled())observe_memory39("take_held",out.bytes());\n        return out;\n    }\n\n    // Reserve before allocating')
 s=once(s,'    bool make_room(size_t incoming, size_t held = 0) {','    bool make_room(size_t incoming, size_t held = 0) {\n        observe_memory39("make_room_before",held,incoming);')
 s=once(s,'        if (bytes() > budget_ - held - incoming) reuse_ = {};','        if (bytes() > budget_ - held - incoming) reuse_ = {};\n        observe_memory39("make_room_after_reusable_release",held,incoming);')
 s=once(s,'        return true;\n    }\n\n    // The parked conversation','        observe_memory39("make_room_after",held,incoming);\n        return true;\n    }\n\n    // The parked conversation')
 s=once(s,'        ++evictions_;','        ++evictions_;\n        observe_memory39("eviction_after");')
 s=once(s,'        const size_t n = image.bytes();','        const size_t n = image.bytes();\n        observe_memory39("put_before",held,0,0,n);')
 s=once(s,'        bytes_ += n;','        bytes_ += n;\n        observe_memory39("put_after",held);');new[path]=s
 path='sycl/src/program/generate.cpp';s=new[path]
 s=once(s,'#include "strata/core/full_cache_observer38.hpp"','#include "strata/core/full_cache_observer38.hpp"\n#include "strata/core/full_cache_memory39.hpp"')
 s=once(s,'            bs[(size_t)b].stop=true;','            bs[(size_t)b].stop=true;\n            strata::core::full_cache_memory39::lifetime("BSTOP_applied",bs[(size_t)b].request_id,bs[(size_t)b].generation,b,0,bs[(size_t)b].p,bs[(size_t)b].ids.size(),bs[(size_t)b].produced,-1,"cancel");')
 s=once(s,'            if (!ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {','            if(strata::core::full_cache38::enabled())for(int a=0;a<A;++a){const auto& sl=bs[(size_t)active[a]];strata::core::full_cache_memory39::lifetime("BSTEP_before",sl.request_id,sl.generation,active[a],uint64_t(bt_windows+1),sl.p,sl.ids.size(),sl.produced,sl.x,"");}\n            if (!ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {')
 s=once(s,'                    ++sl.produced;\n                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y)', '                    ++sl.produced;\n                    strata::core::full_cache_memory39::lifetime("BT_emitted",sl.request_id,sl.generation,b,uint64_t(bt_windows),sl.p,sl.ids.size(),sl.produced,y,"");\n                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y)')
 s=once(s,'                        sl.active = false;\n                        sl.cached = o.prompt_cache > 0 && !sl.img;','                        strata::core::full_cache_memory39::lifetime("BDONE_emitted",sl.request_id,sl.generation,b,uint64_t(bt_windows),sl.p,sl.ids.size(),sl.produced,y,fin,0);\n                        sl.active = false;\n                        sl.cached = o.prompt_cache > 0 && !sl.img;')
 s=once(s,'            strata::core::full_cache38::points("request_complete",checks,o.prompt_cache,tail_ckpt_len);','            strata::core::full_cache38::points("main_body_complete",checks,o.prompt_cache,tail_ckpt_len);')
 s=once(s,'            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>','            strata::core::full_cache_memory39::lifetime("DONE_emitted",req_rid,req_slot_generation,admit_slot,0,p,live.size(),produced_n,-1,finish);\n            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>')
 s=once(s,'                admit_slot = -1;\n            }\n            if (drive.routing', '                strata::core::full_cache_memory39::lifetime("BADM_emitted",req_rid,req_slot_generation,admit_slot,0,p,live.size(),produced_n,-1,finish,cont?1:0);\n                admit_slot = -1;\n            }\n            if (drive.routing')
 s=once(s,'            estimate += stage_estimate;','            estimate += stage_estimate;\n            if(strata::core::full_cache38::enabled())conversations.observe_memory39("park_estimated",held,estimate,strata::core::full_cache_memory39::add(reuse.bytes(),strata::core::full_cache_memory39::local_reuse_bytes(stage_reuse)));')
 s=once(s,'\n                strata::core::SavedConversation image;','\n                if(strata::core::full_cache38::enabled())conversations.observe_memory39("capture_before",held,estimate,strata::core::full_cache_memory39::add(reuse.bytes(),strata::core::full_cache_memory39::local_reuse_bytes(stage_reuse)));\n                strata::core::SavedConversation image;')
 s=once(s,'                        std::move(reuse), &reused_bytes)) return false;','                        std::move(reuse), &reused_bytes)) return false;\n                if(strata::core::full_cache38::enabled())conversations.observe_memory39("capture_main_after",held,estimate,strata::core::full_cache_memory39::add(reuse.bytes(),strata::core::full_cache_memory39::local_reuse_bytes(stage_reuse)),image.bytes());')
 s=once(s,'                    image.stage_images.push_back(std::move(part));','                    image.stage_images.push_back(std::move(part));\n                    if(strata::core::full_cache38::enabled())conversations.observe_memory39("capture_stage_after",held,estimate,strata::core::full_cache_memory39::add(reuse.bytes(),strata::core::full_cache_memory39::local_reuse_bytes(stage_reuse)),image.bytes());')
 s=once(s,'                incoming.reset(); // Running-state/checkpoint copies are no longer needed.','                if(strata::core::full_cache38::enabled())conversations.observe_memory39("restore_before_held_release",incoming->bytes());\n                incoming.reset(); // Running-state/checkpoint copies are no longer needed.\n                conversations.observe_memory39("restore_after_held_release");')
 # Slot snapshot has no transferred retained KV. put_before records its real bytes;
 # this observation binds the estimate before the actual capture path.
 s=once(s,'            if(!conversations.make_room(estimate)||!strata::core::conversation_memory_admit','            conversations.observe_memory39("slot_capture_estimated",0,estimate);\n            if(!conversations.make_room(estimate)||!strata::core::conversation_memory_admit')
 s=once(s,'            strata::core::full_cache38::host_memory();','            strata::core::full_cache38::host_memory();\n            if(strata::core::full_cache38::enabled()){conversations.observe_memory39("main_body_complete");strata::core::full_cache_memory39::chains(strata::core::batch_prefix::chain_bytes(checks),batch_slot_chain_bytes(),strata::core::batch_prefix::settings().chain_bytes,strata::core::batch_prefix::settings().transfer_bytes);strata::core::full_cache_memory39::expert_backing(xcache,0,0);for(size_t k=0;k<stages.size();++k)strata::core::full_cache_memory39::expert_backing(stages[k]->cache,int(k+1),stages[k]->dev);}')
 new[path]=s;return old,new
def patch_bytes():
 old,new=reconstruct();return ''.join(''.join(difflib.unified_diff(old.get(n,'').splitlines(True),new[n].splitlines(True),fromfile='a/'+n if n in old else '/dev/null',tofile='b/'+n))for n in sorted(new)if old.get(n)!=new[n]).encode('ascii')
def build_plan():
 _,new=reconstruct();p=json.loads(BASE.read_bytes())
 if PATCH.read_bytes()!=patch_bytes():raise ValueError('Exact source39 patch changed')
 p['derived_from_plan']={'path':str(BASE.relative_to(ROOT)),'sha256':BASE_SHA}
 p['generation']='source38 plus defaultOFF logical cache memory accounting39'
 p['status']='SOURCE/CPU proposal; no native build/runtime'
 p['patches'].append({'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)})
 p['added_header_payloads'].append({'path':H,'sha256':sha(HEADER),'consumed_include':'strata/core/full_cache_memory39.hpp','resolution':'NEW shared host-only accounting header'})
 p['expected_patched_source_sha256']={n:hashlib.sha256(s.encode()).hexdigest()for n,s in new.items()};p['overlay_files']=sorted(new)
 p['new_source_contracts']['0039']={'default_OFF':True,'parked_reusable_held_incoming_local_reuse_allocated_snapshot_observed':True,'estimated_incoming_overlaps_pending_allocation':True,'logical_peaks_are_sampled_scope_only':True,'whole_process_or_device_peak_qualified':False,'NN_cache_policy_changed':False,'old_source38_runtime_transfer':False}
 p['required_before_model_run'].append('Fresh source39 SDK/ABI/oracle/upload/baseline plus actual memory39 component/peak/overflow/defaultOFF/admission/reuse/stage capture witnesses mandatory; logical samples cannot qualify whole-process or physical device peak.')
 p['actual_memory39_runtime_qualified']=False;return p
