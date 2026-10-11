"""Whole preregistered source40 shared family joins; missing goals stay explicit."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical
import full_cache_shared_runtime_v7 as ctrl

SCENARIOS=('shared','independent','cancellation','eviction','history','stale')

def eviction_victim_binding(proof):
 """Both actual removal classes require the original recollected inventories."""
 logical=proof['actual_recollected_resource_accounting']['actual_logical_source39_accounting']
 ctrl.require(logical['real_checkpoint_victim_observed'] is True and logical['real_parked_victim_observed'] is True and logical['victims'],'Whole eviction family requires actual checkpoint and parked victims; request count alone is insufficient')
 return {'actual_checkpoint_and_parked_victims':logical['victims'],'physical_residency_qualified':False}

def declaration(case):
 from full_cache_shared_phase_contract_v2 import scenario_schedule
 actors=[]
 for scenario in SCENARIOS:
  first=scenario_schedule(case,scenario,0);passes=first['capture_roster']['passes']
  for index in range(len(passes)):
   spec=scenario_schedule(case,scenario,index);actors.append({'name':scenario+'-capture'+str(index),'scenario':scenario,'capture_pass':index,'entire_actor_HTTP_requests':spec['capture_roster']['total_HTTP_requests'],'armed_requests_required':spec['capture_roster']['armed_requests_required'],'selected_actual_RIDs':spec['selected_capture_pass']['selected_actual_RIDs'],'entire_actor_phase_roster':[p['name'] for p in spec['phases']],'cached_state_from_other_actor_allowed':False})
 # Restart is a new independently cold matched shared actor, after full first closure.
 actors.append(dict(next(a for a in actors if a['name']=='shared-capture0'),name='restart-shared-capture0',must_start_after='shared-capture0'))
 eviction=[a for a in actors if a['scenario']=='eviction'];ctrl.require(len(eviction)==2 and all(a['entire_actor_HTTP_requests']==14 and a['armed_requests_required']==12 and len(a['selected_actual_RIDs'])<=6 for a in eviction),'Whole fourteen-HTTP/twelve-armed eviction must retain both complete actor passes')
 return {'source_generation':40,'actual_parent_generation_required':1403,'actors':actors,'serial_persisted_family_required':True,'independent_fresh49_every_actor_required':True,'physical_driver_per_expert_and_transient_peak_required':True,'different_model_live_cache_refusal_separate_from_wrong_disk_fingerprint':True,'all_original_requirements':list(ctrl.MANDATORY),'full_cache_runtime_qualified':False}

def actual_actor_shape(root,declared):
 plan=ctrl.read(Path(root)/'input-plan.snapshot.json');child=ctrl.read(Path(root)/'child/report.json');ctrl.require(plan['scenario']==declared['scenario'] and plan['capture_pass']==declared['capture_pass'] and [r['phase']['name']for r in child['phase_receipts']]==declared['entire_actor_phase_roster'],'Actual entire actor family/phase roster differs');work=[w for r in child['phase_receipts'] for w in r['work']];ctrl.require(len(work)==declared['entire_actor_HTTP_requests'] and len({w['rid']for w in work})==len(work) and {w['rid']for w in work}==set(range(1,len(work)+1)),'Actual every-request owner/metadata roster missing');ctrl.require(plan['scenario_binding']['selected_capture_pass']['selected_actual_RIDs']==declared['selected_actual_RIDs'],'Actual bounded selected RID pass differs');return plan,child

def finalized_binding(plan):
 """ROOT READONLY all families, no replacement of missing physical/model evidence."""
 from validate_full_cache_shared_runtime_v7 import finalized_binding as actor_admit
 from validate_full_cache_shared_batch0_persisted_v7 import finalized_binding as persisted_admit
 from full_cache_shared_restart_v7 import finalized_binding as restart_admit
 case=ctrl.authentic_case();expected=declaration(case);ctrl.require(plan['schema']=='source40-full-cache-suite-v3' and plan['declaration']==expected and plan['engine_plan_sha256']==ctrl.ENGINE_SHA,'Exact entire preregistered source40 family declaration required');items=plan['actor_roots'];ctrl.require(type(items)is list and [r['name']for r in items]==[r['name']for r in expected['actors']] and len({r['root']for r in items})==len(items),'Every declared independent actor root required once')
 reports=[];plans=[];children=[]
 for declared,item in zip(expected['actors'],items):
  root=Path(item['root']).resolve();proof=actor_admit(root,fresh_control_roots=[Path(p) for p in item['fresh_control_roots']]);ctrl.require(proof['all_supplied_comparisons_bitwise_equal'] is True and proof['fresh_control_parent_and_source_binding_unobserved'] is False,'Actual closed independent complete fresh49 comparisons required for every family');ctrl.require(proof['actual_own_API_yielded_ID_HTTP_text_decode']['original_API_yielded_ID_HTTP_text_qualified'] is True,'Actual independent own reply decode required for every actor');actual,child=actual_actor_shape(root,declared);victims=eviction_victim_binding(proof) if declared['scenario']=='eviction' else None;reports.append({'name':item['name'],'root':str(root),'proof':proof,'actual_eviction_victims':victims});plans.append(actual);children.append(child)
 first=plans[0]
 for actual in plans:
  ctrl.require(all(canonical(actual[k])==canonical(first[k]) for k in ('prepared','prepared_sha256','engine_receipt_sha256','baseline_binding','image','cards','pack','authentic_case_sha256','registry_association')),'No family can borrow another source/SDK/model/registry topology')
 byname={r['name']:r for r in items};a=byname['shared-capture0'];b=byname['restart-shared-capture0'];restart=restart_admit(Path(a['root']),Path(b['root']),[Path(p)for p in a['fresh_control_roots']],[Path(p)for p in b['fresh_control_roots']])
 persisted=persisted_admit(Path(plan['serial_persisted_root']),fresh_roots=[Path(p)for p in plan['serial_persisted_fresh_roots']]);ctrl.require(persisted['independent_complete_fresh49_qualified'] is True and persisted['independent_HTTP_ID_text_decode_qualified'] is True,'Actual full persisted/fresh49/independent text family required')
 persisted_plan=ctrl.read(Path(plan['serial_persisted_root'])/'input-plan.snapshot.json');ctrl.require(all(canonical(persisted_plan[k])==canonical(first[k]) for k in ('prepared','prepared_sha256','engine_receipt_sha256','baseline_binding','image','cards','pack','authentic_case_sha256','registry_association')),'Persisted family must use actual same current source/model/topology')
 # All protocol/model/fresh joins are retained. These separate missing scopes
 # cannot be filled by interpreting snapshot byte reservations as residency.
 return {'actual_complete_declared_family_actors':reports,'actual_two_incarnation_restart':restart,'actual_serial_persisted_and_fresh49':persisted,'whole_fourteen_HTTP_twelve_armed_eviction_actors':True,'cached_state_transferred_between_actor_processes':False,'all_original_requirements':expected['all_original_requirements'],'different_model_live_cache_refusal_qualified':False,'physical_driver_per_expert_residency_qualified':False,'whole_process_transient_peak_qualified':False,'remaining_separate_required_scopes':['actual changed-model live-cache refusal','actual physical host/device/per-expert attribution','actual whole-actor/transient peak coverage'],'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'latency_qualified':False}
