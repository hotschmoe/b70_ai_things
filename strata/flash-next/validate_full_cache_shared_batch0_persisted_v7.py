"""Readonly complete serial persisted family; independent fresh49 required."""
import hashlib,json
from pathlib import Path
import full_cache_shared_batch0_persisted_v7 as ctrl
import qualify_full_cache_shared_batch0_persisted_v7 as wrapper
import run_full_cache_shared_batch0_persisted_v7 as run
from full_cache_shared_closed_source_v7 import admit
from full_cache_shared_batch0_pcl_v7 import full49
from full_cache_shared_persisted_lineage_v7 import session_binding,saved_files,refusal_state_unchanged
from full_cache_shared_raw49_v2 import compare49
from serial37_canonical_json_v3 import canonical


def finalized_binding(root,fresh_roots=None):
 root=Path(root).resolve();parent,plan,child=admit(root,ctrl,Path(wrapper.__file__));before=ctrl.manifest_binding(plan);directory=root/'child';events=run.common.events(directory/'api-native-trace.jsonl');ctrl.require(ctrl.read(directory/'launch.command.json')==run.command_recipe(plan,directory,parent['child_pid']),'Actual complete purpose Docker launch differs')
 from full_cache_shared_launch_binding_v7 import binding as launch_binding
 launch_binding(directory,child,plan)
 from full_cache_shared_actor_retirement_v7 import saved_binding as retirement_binding
 retirement_binding(root/'child',child,ctrl.read(root/'child/launch.command.json')[ctrl.read(root/'child/launch.command.json').index('--name')+1],ctrl.read)
 cfg=ctrl.read(Path(plan['prepared'])/'server-config.json');cfg.update(args=plan['args'],env=plan['env'],parallel=1,port=plan['port'],model_name='hotschmoe-dd',aliases=[plan['research_alias']],log='/results/server-engine.log',slot_save_path='/results/sessions');ctrl.require(cfg==ctrl.read(directory/'server-config.json'),'Actual original serial frontend config differs');models=ctrl.read(directory/'models.json');identity=ctrl.sha(directory/'artifact-identity.json');ctrl.require([r['id']for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==identity for r in models['data']),'Actual original model alias/artifact identity differs')
 ctrl.require([r['phase']for r in child['request_receipts']]==['prime','after_wrong','after_valid'] and [r['phase']for r in child['session_receipts']]==['save','wrong_restore','valid_restore'],'Entire original cold/save/refuse/repeat/validrestore/repeat roster required');saved={r['phase']:r for r in child['request_receipts']+child['session_receipts']};numbers={name:i for i,name in enumerate(plan['schedule']['preregistered_phase_order'],1)};numeric={};session_ends={};actual_owner=None
 for name in plan['schedule']['preregistered_phase_order']:
  record=saved[name];path=directory/'phases'/name;ack=record['producer_ack'];ctrl.require(ctrl.read(path/'receipt.json')==record and record['spec']==plan['schedule'][name] and ack['index']==numbers[name] and ack['name']==name,'Actual original phase recipe/receipt changed');markers=[e for e in events if e['kind']=='fullcache_phase_begin' and e['index']==numbers[name] and e['name']==name];ctrl.require(len(markers)==1 and all(markers[0][k]==v for k,v in ack.items()),'Original phase marker/ACK association differs');current=[e for e in events if ack['sequence']<e['sequence']<=record['end_sequence']]
  if name in ('prime','after_wrong','after_valid'):
   begins=[e for e in current if e['kind']=='engine_begin'];ends=[e for e in current if e['kind']=='engine_end'];ctrl.require(len(begins)==len(ends)==1 and begins[0]==record['native_begin'] and ends[0]==record['native_end'],'Actual original sole FIFO begin/end differs');from full_cache_shared_client_recollection_v7 import recollect as client_recollect
   client_recollect(path/'client',record['client'],[record['spec']['row']['messages']],[1],{'strata_fresh':bool(record['spec']['fresh'])},plan['research_alias'])
   spec=record['spec'];expected={k:spec[k]for k in ('fresh','pin','expected_reused')};expected['ids']=spec['row']['ids'];result=full49(current,begins[0],ends[0],expected,directory/'captures',ctrl.expected_stage_ranges(plan['args']));ctrl.require(canonical(result)==canonical(record['numeric']),'Original actual full49/PCL/HTTP source recollection differs');numeric[name]=result
   from full_cache_shared_batch0_decode_v7 import finalized_binding as own_decode
   ctrl.require(own_decode(root,directory/'serial-decodes'/name)==record['independent_ID_text_decode'],'Original independent owned ID/text decode changed')
   if actual_owner is None:actual_owner={'engine_pid':begins[0]['engine_pid'],'engine_generation':begins[0]['engine_generation']}
   ctrl.require(actual_owner=={'engine_pid':begins[0]['engine_pid'],'engine_generation':begins[0]['engine_generation']},'Persisted actor native incarnation changed')
  else:
   http=ctrl.read(path/'HTTP.receipt.json');raw=(path/'HTTP.body.json').read_bytes();ctrl.require(http==record['HTTP'] and json.loads(raw)==http['body'] and hashlib.sha256(raw).hexdigest()==http['raw_body_sha256'],'Actual original persisted raw HTTP response differs');binding=session_binding(current,http,record['spec'],actual_owner);ctrl.require(binding==record['native_HTTP_binding'],'Original actual native/session/HTTP association differs');ends=[e for e in current if e['kind']=='session_end'];ctrl.require(len(ends)==1,'Actual original session end absent');session_ends[name]=ends[0]
 negative=ctrl.read(directory/'negative-file-derivation.json');ctrl.require(negative==child['negative_derivation'] and saved_files(directory,negative,plan['schedule']['save']['expected_saved_tokens'],numeric['prime']['input_ids'])==child['saved_files_binding'],'Original saved and model-only negative files differ');ctrl.require(refusal_state_unchanged(numeric['prime'],numeric['after_wrong'],session_ends['wrong_restore'],numeric['prime']['input_ids'])==child['refusal_unchanged49'],'Actual own after-refusal full49 changed');ctrl.require(compare49(numeric['prime'],numeric['after_valid'])==child['valid_restore49'] and child['valid_restore49']['all49_bitwise_equal'] is True,'Actual own valid-restored full49 changed')
 status=ctrl.read(directory/'api-native-trace.jsonl.buffered-status.json');ctrl.require(status==child['buffered_status'] and status['passed'] is True and status['closed'] is True and status['fullcache_phase_thread_retired'] is True and status['fullcache_control_thread_retired'] is True and status['fullcache_phase_errors']==[] and status['fullcache_control_errors']==[],'Original serial marker/control/buffer EOF changed');closed=[e for e in events if e['kind']=='engine_close'];ctrl.require(len(closed)==1 and closed[0]['exit_code']==0 and closed[0]['error'] is None,'Actual normal serial native exit missing')
 comparisons=[]
 if fresh_roots is not None:
  # This requires purpose-prepared fresh roots; arbitrary matrices/flags are not admission.
  from validate_full_cache_shared_batch0_fresh40_v7 import complete_controls
  controls,control_bindings=complete_controls(root,fresh_roots)
  for name,result in numeric.items():
   from full_cache_shared_history_v2 import digest
   key=digest(result['input_ids']);ctrl.require(key in controls,'Actual independently owned fresh49 control missing');comparisons.append(compare49(result,controls[key]))
 ctrl.require(canonical(ctrl.manifest_binding(plan))==canonical(before),'Current source40/purpose baseline changed during reader')
 return {'actual_all_serial_persisted_phases_recollected':True,'actual_full49_requests':3,'actual_session_operations':3,'actual_own_before_after_state_control_49_bitwise':True,'independent_fresh49_comparisons':comparisons,'independent_complete_fresh49_qualified':bool(comparisons) and len(comparisons)==3 and all(r['all49_bitwise_equal'] for r in comparisons),'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'physical_memory_or_expert_residency_qualified':False,'independent_HTTP_ID_text_decode_qualified':True}
