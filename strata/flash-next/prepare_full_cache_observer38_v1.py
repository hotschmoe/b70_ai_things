"""Source38 CPU reconstruction only; no compilation, model or GPU execution."""
import difflib,hashlib,json
from pathlib import Path
import prepare_host_critical_path36_v1 as trace36
import prepare_batch_stage_stamp37_v1 as stamp37
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=HERE/'hosttrace36-batchstamp37-engine-build-plan-v1.json'
BASE_SHA='2e940d51c61abe5366526b926f89dcf9f42bf778feee5d13c9b72aefc2785ace'
HEADER=HERE/'full_cache_observer38.hpp';H='include/strata/core/full_cache_observer38.hpp'
PATCH=HERE/'patches/0038-default-off-selected-rids-cache-victim-memory-observer.patch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(text,old,new):
 if text.count(old)!=1:raise ValueError('Exact source38 hook anchor changed: '+old[:80])
 return text.replace(old,new)
def reconstruct():
 if sha(BASE)!=BASE_SHA:raise ValueError('Frozen source37 plan changed')
 source35=json.loads(stamp37.BASE.read_bytes());base={n:(stamp37.SDK/n).read_text() for n in source35['expected_patched_source_sha256']}
 for n,data in base.items():
  if hashlib.sha256(data.encode()).hexdigest()!=source35['expected_patched_source_sha256'][n]:raise ValueError('Exact base35 code changed '+n)
 _,overlay=trace36.reconstruct();base.update(overlay);base[stamp37.V]=once(base[stamp37.V],stamp37.ANCHOR,stamp37.ANCHOR.replace('        return true;',stamp37.HOOK+'        return true;'))
 plan=json.loads(BASE.read_bytes())
 for name,data in base.items():
  if hashlib.sha256(data.encode()).hexdigest()!=plan['expected_patched_source_sha256'][name]:raise ValueError('Exact source37 byte closure differs '+name)
 new=dict(base);new[H]=HEADER.read_text()
 c='sycl/include/strata/core/batch_fidelity_contract.hpp';s=new[c]
 s=once(s,'phase>2','phase>3');s=once(s,'bool admission=false,later=false,migration=false,closed_cancel=false;','bool admission=false,later=false,migration=false,closed_cancel=false,direct_done=false;')
 s=once(s,'e->id.slot=slot;e->id.slotgen=slotgen;e->id.phase=1;return e->id;','e->id.slot=slot;e->id.slotgen=slotgen;if(e->id.phase==3){e->id.prompt=prompt;e->admission=false;e->later=false;e->migration=false;e->closed_cancel=false;}e->id.phase=1;return e->id;')
 anchor=' void closed(uint64_t rid,uint64_t slotgen,int slot,bool cancelled)'
 direct=''' identity direct(uint64_t rid,int64_t prompt,uint64_t enginegen){
  if(!rid||!enginegen||prompt<1||find(rid)||count>=REQUESTS)throw std::invalid_argument("batch direct actual identity/bound");
  auto& e=requests[count++];e.id={rid,count,0,ROWS,prompt,0,0};e.id.enginegen=enginegen;e.id.phase=3;return e.id;
 }
'''
 s=once(s,anchor,direct+anchor)
 s=once(s,'  if(phase==2){if(id.phase!=2', '  if(phase==3){if(id.phase!=3||e->id.phase!=3||id.slot!=ROWS||id.slotgen!=0||id.event!=0||id.prompt!=e->id.prompt)throw std::invalid_argument("batch observer stale direct GEN identity");return !e->direct_done;}\n  if(phase==2){if(id.phase!=2')
 s=once(s,'else e->migration=true;','else if(phase==2)e->migration=true;else e->direct_done=true;');new[c]=s
 b='sycl/include/strata/core/batch_fidelity_observer.hpp';s=new[b]
 s=once(s,'#include "strata/core/batch_fidelity_contract.hpp"','#include "strata/core/batch_fidelity_contract.hpp"\n#include "strata/core/full_cache_observer38.hpp"')
 s=once(s,'admitting()={};if(!armed()||!rid)return {};if(slot<0){','admitting()={};if(!armed()||!rid)return {};if(full_cache38::enabled()&&!full_cache38::selected(rid))return {};if(slot<0&&full_cache38::enabled()&&!records().find(rid)){auto id=records().direct(rid,prompt,engine_generation());admitting()=id;std::fprintf(stderr,"SBF direct_gen_begin pid=%ld rid=%llu enginegen=%llu requestgen=%llu slot=6 slotgen=0 prompt=%lld event=0 normal_dispatch=GEN observed_only=1\\n",long(getpid()),(unsigned long long)rid,(unsigned long long)id.enginegen,(unsigned long long)id.generation,(long long)prompt);return id;}if(slot<0){')
 s=once(s,'phase==2?ROWS+1:ROWS','phase>=2?ROWS+1:ROWS');s=once(s,'phase==1?id.event!=0:id.event==0','(phase==1||phase==3)?id.event!=0:id.event==0')
 s=once(s,'phase==2?"solo_migration_residual":phase==1?"admission_residual":"batch_step_residual"','phase==3?"direct_gen_residual":phase==2?"solo_migration_residual":phase==1?"admission_residual":"batch_step_residual"')
 s=once(s,'phase==2?"solo_migration_logits_before_sampler":phase==1?"admission_logits_before_sampler":"batch_step_logits_before_sampler"','phase==3?"direct_gen_logits_before_sampler":phase==2?"solo_migration_logits_before_sampler":phase==1?"admission_logits_before_sampler":"batch_step_logits_before_sampler"');new[b]=s
 v='sycl/src/core/verify.cpp';s=new[v];old='batch_phase_==2?batch_migration_exec_[ar_off_?1:0]:batch_admission_exec_[ar_off_?1:0]'
 if s.count(old)!=2:raise ValueError('Exact two actual observer graph route bindings changed')
 s=s.replace(old,'batch_phase_==3?batch_direct_exec_[ar_off_?1:0]:batch_phase_==2?batch_migration_exec_[ar_off_?1:0]:batch_admission_exec_[ar_off_?1:0]')
 s=once(s,'batch_phase_=(n==1&&ids[0].phase==2)?2:(admission?1:0);','batch_phase_=(n==1&&ids[0].phase==3)?3:(n==1&&ids[0].phase==2)?2:(admission?1:0);')
 s=once(s,'batch_phase_==2?"solo_migration_target_verify":"admission_target_verify"','batch_phase_==3?"direct_gen_target_verify":batch_phase_==2?"solo_migration_target_verify":"admission_target_verify"')
 s=once(s,'for(auto& p:batch_migration_exec_){if(p)delete p;p=nullptr;}','for(auto& p:batch_migration_exec_){if(p)delete p;p=nullptr;}for(auto& p:batch_direct_exec_){if(p)delete p;p=nullptr;}');new[v]=s
 v='sycl/include/strata/core/verify.hpp';new[v]=once(new[v],'batch_migration_exec_[2]={};','batch_migration_exec_[2]={},batch_direct_exec_[2]={};')
 g='sycl/src/program/generate.cpp';s=new[g]
 s=once(s,'#include <cstdio>','#include <cstdio>\n#include "strata/core/full_cache_observer38.hpp"')
 anchor='    if(strata::core::batch_fidelity::settings().enabled && (!strict_batch_identity'
 s=once(s,anchor,'    if(strata::core::full_cache38::enabled() && !strata::core::batch_fidelity::settings().enabled){std::fprintf(stderr,"fullcache38 requires existing strict batch diagnostic profile\\n");return 2;}\n'+anchor)
 s=once(s,'            const auto batch_admission_id=strata::core::batch_fidelity::begin','            strata::core::full_cache38::begin(req_rid,req_slot_generation,admit_slot,ids,req_fresh,req_pin_present,req_pin);\n            const auto batch_admission_id=strata::core::batch_fidelity::begin')
 old='admission_observed_id.phase==2?2:1'
 if s.count(old)!=2:raise ValueError('Exact selected first-window completion phase changed')
 s=s.replace(old,'admission_observed_id.phase==3?3:admission_observed_id.phase==2?2:1')
 s=once(s,'                checks.erase(checks.begin() + (std::ptrdiff_t) victim);','                strata::core::full_cache38::points("before_eviction",checks,o.prompt_cache,tail_ckpt_len,int64_t(victim));\n                checks.erase(checks.begin() + (std::ptrdiff_t) victim);\n                strata::core::full_cache38::points("after_eviction",checks,o.prompt_cache,tail_ckpt_len);')
 s=once(s,'            checks.push_back(std::move(c));','            checks.push_back(std::move(c));\n            strata::core::full_cache38::points("checkpoint_saved",checks,o.prompt_cache,tail_ckpt_len);')
 s=once(s,'            strata::core::prefix_lifecycle::state("committed_live",live,!cancelled,live_ok,cancelled?"prefill":std::strcmp(finish,"cancel")==0?"decode":"complete",finish);','            strata::core::prefix_lifecycle::state("committed_live",live,!cancelled,live_ok,cancelled?"prefill":std::strcmp(finish,"cancel")==0?"decode":"complete",finish);\n            strata::core::full_cache38::work(resume,read_from,reread_to,cancelled?pp_reached:n,n,produced_n,cancelled||std::strcmp(finish,"cancel")==0,finish);\n            strata::core::full_cache38::points("request_complete",checks,o.prompt_cache,tail_ckpt_len);\n            strata::core::full_cache38::host_memory();\n            if(strata::core::full_cache38::enabled()){strata::core::full_cache38::experts(xcache,0,0,0,stages.empty()?g.n_layers:stages[0]->lb,g.n_expert);for(size_t k=0;k<stages.size();++k){const auto& st=stages[k];strata::core::full_cache38::experts(st->cache,int(k+1),st->dev,st->lb,st->le,g.n_expert);}}')
 new[g]=s
 # Parked LRU witness records the actual already-selected victim and preserves
 # snapshot identities across the existing deque erase. No policy recalculation.
 c='include/strata/core/conversation_cache.hpp';s=new[c];s=once(s,'#include "strata/core/prefix_lifecycle.hpp"','#include "strata/core/prefix_lifecycle.hpp"\n#include "strata/core/full_cache_observer38.hpp"')
 s=once(s,'        entries_.push_back(std::move(image));','        entries_.push_back(std::move(image));\n        full_cache38::admit(&entries_.back());')
 s=once(s,'        bytes_ -= victim->bytes();','        full_cache38::parked("before_eviction",entries_,budget_,bytes_,slots_,int64_t(victim-entries_.begin()));\n        const auto full_cache38_registry_before=full_cache38::capture(entries_);\n        bytes_ -= victim->bytes();')
 s=once(s,'        entries_.erase(victim);','        entries_.erase(victim);\n        full_cache38::rebind(entries_,removed,full_cache38_registry_before);\n        full_cache38::parked("after_eviction",entries_,budget_,bytes_,slots_);')
 s=once(s,'        bytes_ += n;','        bytes_ += n;\n        full_cache38::parked("admit",entries_,budget_,bytes_,slots_);');new[c]=s
 s=once(s,'        SavedConversation out = std::move(entries_.at(index));','        full_cache38::parked("before_take",entries_,budget_,bytes_,slots_,int64_t(index));\n        const auto full_cache38_take_before=full_cache38::capture(entries_);\n        SavedConversation out = std::move(entries_.at(index));')
 s=once(s,'        prefix_lifecycle::rebind_after_erase(entries_,index,registry_before);','        prefix_lifecycle::rebind_after_erase(entries_,index,registry_before);\n        full_cache38::rebind(entries_,index,full_cache38_take_before);\n        full_cache38::parked("after_take",entries_,budget_,bytes_,slots_);')
 s=once(s,'                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);','                full_cache38::parked("before_superseded",entries_,budget_,bytes_+e.bytes(),slots_,int64_t(i));\n                const auto full_cache38_superseded_before=full_cache38::capture(entries_);\n                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);')
 s=once(s,'                prefix_lifecycle::rebind_after_erase(entries_,i,registry_before);','                prefix_lifecycle::rebind_after_erase(entries_,i,registry_before);\n                full_cache38::rebind(entries_,i,full_cache38_superseded_before);\n                full_cache38::parked("after_superseded",entries_,budget_,bytes_,slots_);');new[c]=s
 return base,new
def patch_bytes():
 old,new=reconstruct();return ''.join(''.join(difflib.unified_diff(old.get(n,'').splitlines(True),new[n].splitlines(True),fromfile=('a/'+n if n in old else '/dev/null'),tofile='b/'+n)) for n in sorted(new) if old.get(n)!=new[n]).encode('ascii')
def build_plan():
 old,new=reconstruct();p=json.loads(BASE.read_bytes())
 if PATCH.read_bytes()!=patch_bytes():raise ValueError('Actual source38 patch differs from reconstructed bytes')
 p['derived_from_plan']={'path':str(BASE.relative_to(ROOT)),'sha256':BASE_SHA};p['generation']='source37 plus defaultOFF selected actual RIDs/directGEN/checkpoint victim/memory observer38';p['status']='SOURCE/CPU proposal; not compiled or qualified';p['patches'].append({'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)});p['expected_patched_source_sha256']={n:hashlib.sha256(b.encode()).hexdigest() for n,b in new.items()};p['added_header_payloads'].append({'path':H,'sha256':sha(HEADER),'consumed_include':'strata/core/full_cache_observer38.hpp','resolution':'NEW shared host-only diagnostic header'})
 p['new_source_contracts']['0038']={'default_OFF':True,'selected_actual_RIDs_maximum':6,'raw_byte_limit_unchanged':64<<20,'all_request_metadata':True,'real_checkpoint_and_parked_victim_observer':True,'truthful_direct_GEN_phase3':True,'NN_math_and_scheduling_changed':False,'old_source37_runtime_proof_transfer':False,'physical_device_residency_qualified':False}
 p['required_before_model_run'].append('Fresh source38 SDK eightELFs/current390/upload/new4/pages/percard+compiledpair health and strong new C1 generation mandatory. DefaultOFF equivalence and selected RID/directGEN/GEN1/solo plus actual point/LRU victim/partialwork and hostmemory fields require genuine native requalification. Old source37 proofs are historical only.')
 p['overlay_files']=sorted(new)
 p['actual_full_cache38_runtime_qualified']=False;return p
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 if args.output.exists():raise ValueError('New output required')
 args.output.write_text(json.dumps(build_plan(),indent=2)+'\n')
 print(json.dumps({'plan_sha256':sha(args.output),'actual_native_compile':False,'actual_GPU_run':False}))
