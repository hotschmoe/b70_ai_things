"""Two actual source40 actor incarnations; no cached state crosses processes."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from full_cache_shared_history_v2 import require

def empty_actor(parent,plan,child):
 require(parent['passed'] is True and parent['errors']==[] and child['collection_and_teardown_passed'] is True and child['removed'] is True and child['error'] is None,'Independent actual actor must finish normally')
 require(plan['lane']=='source40' and type(parent['child_pid'])is int and parent['child_pid']==child['producer_pid'] and type(child['producer_pid'])is int and child['producer_pid']>0,'Actual fresh source40 host controller required')
 rows=child['phase_receipts'];require(rows and rows[0]['phase']['name']=='warm' and len(rows[0]['work'])==2 and rows[0]['phase']['policy']=={'strata_fresh':True},'New actor must start with its own explicit cold warm cohort')
 require({w['rid'] for w in rows[0]['work']}=={1,2} and all(type(w['rid'])is int and type(w['actual_reused'])is int and type(w['actual_new_prompt_tokens'])is int and w['actual_reused']==0 and w['actual_new_prompt_tokens']==len(w['input_ids']) and w['completed_prompt_read'] is True for w in rows[0]['work']),'Actual new process counter/cold whole-prompt observations missing')
 prime=[r for r in rows if r['phase']['name']=='prime0'];require(len(prime)==1 and len(prime[0]['work'])==1 and type(prime[0]['work'][0]['actual_reused'])is int and type(prime[0]['work'][0]['actual_new_prompt_tokens'])is int and prime[0]['work'][0]['actual_reused']==0 and prime[0]['work'][0]['actual_new_prompt_tokens']==len(prime[0]['work'][0]['input_ids']),'Actual shared boundary must be re-established cold in new actor')
 return {'host_controller_pid':child['producer_pid'],'actual_warm_RIDs':[1,2],'actual_new_process_prompt_cold_observed':True,'whole_process_initial_state_directly_sampled':False,'cross_actor_state_borrowed':False,'full_cache_runtime_qualified':False}

def finalized_binding(first,second,first_controls,second_controls):
 """ROOT READONLY: admit both complete source/actual full49 control families."""
 from validate_full_cache_shared_runtime_v6 import finalized_binding as admit
 import full_cache_shared_runtime_v6 as ctrl
 a=Path(first).resolve();b=Path(second).resolve();require(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a),'Distinct independently owned actor roots required')
 proofs=[admit(root,fresh_control_roots=controls) for root,controls in ((a,first_controls),(b,second_controls))]
 require(all(p['all_supplied_comparisons_bitwise_equal'] is True and p['fresh_control_parent_and_source_binding_unobserved'] is False for p in proofs),'Each actor needs its own independently owned complete fresh49 comparisons')
 plans=[ctrl.read(root/'input-plan.snapshot.json') for root in (a,b)];parents=[ctrl.read(root/'parent-qualification.json') for root in (a,b)];children=[ctrl.read(root/'child/report.json') for root in (a,b)]
 require(all(canonical(plans[0][key])==canonical(plans[1][key]) for key in ('lane','args','env','cards','image','pack','authentic_case_sha256','schedule','scenario_binding','engine_receipt_sha256','baseline_binding')),'Restart must keep actual source/config/input/capture recipe matched')
 views=[empty_actor(p,v,c) for p,v,c in zip(parents,plans,children)]
 # Native PIDs can repeat inside distinct Docker PID namespaces. Root host
 # controller PID plus the actual full Docker container ID identifies actors.
 inspections=[ctrl.read(root/'child/owned-terminal-inspection.json') for root in (a,b)]
 require(views[0]['host_controller_pid']!=views[1]['host_controller_pid'] and all(type(i['Id'])is str and i['Id'] for i in inspections) and inspections[0]['Id']!=inspections[1]['Id'],'Actual distinct host/container incarnations required')
 require(parents[0]['finished_epoch']<=children[1]['started_epoch'],'First actor must fully close before restarted actor begins')
 return {'actual_source40_restart_and_reestablishment_observed':True,'actor_views':views,'actual_container_IDs':[i['Id'] for i in inspections],'separate_owned_fresh49_bindings':proofs,'persistent_session_state_restored':False,'full_cache_runtime_qualified':False,'physical_memory_qualified':False}
