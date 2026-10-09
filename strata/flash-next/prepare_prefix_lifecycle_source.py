#!/usr/bin/env python3
"""Prepare draft0019 in a new source overlay; never compile or edit base source."""
import difflib,hashlib,json,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/source')
FILES=['include/strata/core/conversation_cache.hpp','sycl/src/program/generate.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp']

def replace(s,old,new):
 assert s.count(old)==1,(old[:100],s.count(old))
 return s.replace(old,new)

def main():
 out=Path(tempfile.mkdtemp(prefix='strata-prefix-lifecycle0019-',dir='/mnt/vm_8tb/b70/build'));source=out/'source';original={}
 for name in FILES:
  raw=(BASE/name).read_text();original[name]=raw;p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(raw)
 header='include/strata/core/prefix_lifecycle.hpp';p=source/header;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/'strata/flash-next/prefix_lifecycle_observer.hpp').read_bytes())
 name=FILES[0];s=original[name];s=replace(s,'#include "strata/core/conversation_buffer.hpp"','#include "strata/core/conversation_buffer.hpp"\n#include "strata/core/prefix_lifecycle.hpp"')
 s=replace(s,'        SavedConversation out = std::move(entries_.at(index));','        const auto diagnostic = prefix_lifecycle::describe(entries_.at(index));\n        const auto registry_before = prefix_lifecycle::capture_instances(entries_);\n        SavedConversation out = std::move(entries_.at(index));')
 s=replace(s,'        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);','        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);\n        prefix_lifecycle::rebind_after_erase(entries_,index,registry_before);')
 s=replace(s,'        return out;\n    }\n\n    // Reserve before allocating', '        if (prefix_lifecycle::active()) prefix_lifecycle::cache("take",diagnostic,bytes(),size(),budget_,slots_,out.bytes());\n        return out;\n    }\n\n    // Reserve before allocating')
 s=replace(s,'        bytes_ -= victim->bytes();\n        entries_.erase(victim);\n        ++evictions_;','        const auto diagnostic = prefix_lifecycle::describe(*victim);\n        bytes_ -= victim->bytes();\n        const auto registry_before = prefix_lifecycle::capture_instances(entries_);\n        const size_t removed = size_t(victim-entries_.begin());\n        entries_.erase(victim);\n        prefix_lifecycle::rebind_after_erase(entries_,removed,registry_before);\n        ++evictions_;\n        if (prefix_lifecycle::active()) prefix_lifecycle::cache("evict",diagnostic,bytes(),size(),budget_,slots_);')
 s=replace(s,'                bytes_ -= e.bytes();\n                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);','                const auto diagnostic = prefix_lifecycle::describe(e);\n                bytes_ -= e.bytes();\n                const auto registry_before = prefix_lifecycle::capture_instances(entries_);\n                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);\n                prefix_lifecycle::rebind_after_erase(entries_,i,registry_before);\n                if (prefix_lifecycle::active()) prefix_lifecycle::cache("superseded",diagnostic,bytes(),size(),budget_,slots_);')
 s=replace(s,'        const size_t n = image.bytes();\n        if (!enabled() || held > budget_ || n > budget_ - held) return false;', '        const size_t n = image.bytes();\n        auto diagnostic = prefix_lifecycle::describe(image);\n        if (!enabled() || held > budget_ || n > budget_ - held) {\n            if (prefix_lifecycle::active()) prefix_lifecycle::cache("refused_budget",diagnostic,bytes(),size(),budget_,slots_,held);\n            return false;\n        }')
 s=replace(s,'        if (!make_room(n, held)) return false;\n        entries_.push_back(std::move(image));\n        bytes_ += n;','        if (!make_room(n, held)) {\n            if (prefix_lifecycle::active()) prefix_lifecycle::cache("refused_room",diagnostic,bytes(),size(),budget_,slots_,held);\n            return false;\n        }\n        entries_.push_back(std::move(image));\n        bytes_ += n;\n        if (prefix_lifecycle::active()) diagnostic.instance=prefix_lifecycle::admit_instance(&entries_.back());\n        if (prefix_lifecycle::active()) prefix_lifecycle::cache("admit",diagnostic,bytes(),size(),budget_,slots_,held);')
 (source/name).write_text(s)
 name=FILES[1];s=original[name];s='#include "strata/core/prefix_lifecycle.hpp"\n'+s
 s=replace(s,'            prefix_diag.begin(ids,int64_t(stages.size()+1));','            prefix_diag.begin(ids,int64_t(stages.size()+1));\n            strata::core::prefix_lifecycle::begin(ids,prefix_diag.active && o.batch==0 && !use_mtp && o.pipeline_windows==0 && !geni);')
 # Exact committed prefix, including source-confirmed EOS-not-consumed behavior.
 anchor='            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;'
 s=replace(s,anchor,'            strata::core::prefix_lifecycle::state("committed_live",live,!cancelled,live_ok,cancelled?"prefill":std::strcmp(finish,"cancel")==0?"decode":"complete",finish);\n'+anchor)
 # Explicit skipped admissions before put: describe the still-existing exact source prefix.
 s=replace(s,'            if (!conversations.make_room(estimate, held)) {','            if (!conversations.make_room(estimate, held)) {\n                if (strata::core::prefix_lifecycle::active()) {\n                    strata::core::prefix_lifecycle::Snapshot d;strata::core::prefix_lifecycle::token_digest(live,d.live_key);d.tokens=live.size();d.bytes=estimate;\n                    strata::core::prefix_lifecycle::cache("skip_estimated_budget",d,conversations.bytes(),conversations.size(),size_t(o.conversation_cache_mib)<<20,size_t(o.conversation_cache_slots),held);\n                }')
 s=replace(s,'                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission;', '                    if (strata::core::prefix_lifecycle::active()) {\n                        strata::core::prefix_lifecycle::Snapshot d;strata::core::prefix_lifecycle::token_digest(live,d.live_key);d.tokens=live.size();d.bytes=estimate;\n                        strata::core::prefix_lifecycle::cache("skip_pre_ram",d,conversations.bytes(),conversations.size(),size_t(o.conversation_cache_mib)<<20,size_t(o.conversation_cache_slots),held);\n                    }\n                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission;')
 s=replace(s,'                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)', '                    if (strata::core::prefix_lifecycle::active()) strata::core::prefix_lifecycle::cache("skip_post_ram",strata::core::prefix_lifecycle::describe(image),conversations.bytes(),conversations.size(),size_t(o.conversation_cache_mib)<<20,size_t(o.conversation_cache_slots),held);\n                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)')
 (source/name).write_text(s)
 name=FILES[2];s=original[name];s='#include "strata/core/prefix_lifecycle.hpp"\n'+s
 start=s.index('bool Verifier::run(int T,');end=s.index('\nvoid Verifier::set_plan_slot',start);piece=s[start:end]
 piece=replace(piece,'    refresh_ar();','    prefix_lifecycle::stage("verify_body",device_,lb_,le_,pos0,pos0+T,false);\n    refresh_ar();')
 piece=replace(piece,'    if (le_ < g.n_layers) {   // a layer split\'s earlier stage: the hand-off is written (synced above)','    prefix_lifecycle::stage("verify_body",device_,lb_,le_,pos0,pos0+T,true);\n    if (le_ < g.n_layers) {   // a layer split\'s earlier stage: the hand-off is written (synced above)')
 s=s[:start]+piece+s[end:];(source/name).write_text(s)
 name=FILES[3];s=original[name];s='#include "strata/core/prefix_lifecycle.hpp"\n'+s
 start=s.index('bool Prefill::run_impl(');end=s.index('\nbool Prefill::drain_pipeline',start);piece=s[start:end]
 piece=replace(piece,'    const int64_t LB = stage_lb_, LE = stage_le_;','    const int64_t LB = stage_lb_, LE = stage_le_;\n    int64_t lifecycle_completed_rows=0;\n    core::prefix_lifecycle::stage("prefill_body",m.device,LB,LE,pos0,pos0+n,false);')
 piece=replace(piece,'        stats_.tokens += T;','        lifecycle_completed_rows+=T;\n        stats_.tokens += T;')
 last=piece.rindex('    return true;');piece=piece[:last]+'    core::prefix_lifecycle::stage("prefill_body",m.device,LB,LE,pos0,pos0+lifecycle_completed_rows,lifecycle_completed_rows==n);\n'+piece[last:]
 s=s[:start]+piece+s[end:];(source/name).write_text(s)
 patch=[]
 for name in [*FILES,header]:
  before=original.get(name,'');after=(source/name).read_text()
  patch+=difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name if before else '/dev/null',tofile='b/'+name,n=5)
 path=ROOT/'strata/flash-next/patches/0019-sycl-prefix-cache-lifecycle-telemetry.patch';path.write_text(''.join(patch))
 receipt={'scope':'source preparation only; no GPU/compiler/runtime; partial admission provenance explicitly unobserved','base_source':str(BASE),'overlay':str(source),'patch':str(path.relative_to(ROOT)),'patch_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'base_files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in original.items()},'patched_files':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in [*FILES,header]},'state_proof':False,'all_admission_refusals_observed':False,'stage_completion':'body completion after existing source waits, not commit/partial-work proof'}
 (out/'source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(ROOT/'strata/flash-next/prefix-lifecycle-source-plan.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('OVERLAY',source);print('PATCH_SHA',receipt['patch_sha256'])
if __name__=='__main__':main()
