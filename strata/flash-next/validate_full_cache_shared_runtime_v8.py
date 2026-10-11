"""Read-only closed phase/source admission; absent controls never become PASS."""
import hashlib,json,math
from pathlib import Path
import full_cache_shared_runtime_v8 as ctrl
import run_full_cache_shared_runtime_v8 as run
from full_cache_shared_phase_contract_v2 import calls_and_work,phase_events,full_scope_status
from full_cache_shared_terminal_v2 import associate_events
from full_cache_shared_native_lifetime_v5 import recollect as native_lifetime
from api_journal_exact_admission_v5 import journal_gate
from validate_batch_api_cache_positive_buffered_v5 import exact_health_gate
from source_page_watchdog_v3 import KNOWN_PAGES
def finalized_binding(directory,fresh_groups=None,fresh_control_roots=None):
 ctrl.require(fresh_groups is None,'Caller-supplied fresh arrays are not actual owned/source-qualified fresh49 controls; future explicit control admission required')
 root=Path(directory).resolve();from full_cache_shared_closed_source_v8 import admit
 parent,plan,child=admit(root,ctrl,Path(__file__).with_name('qualify_full_cache_shared_runtime_v8.py'));before=ctrl.manifest_binding(plan)
 ctrl.require(ctrl.read(root/'child/launch.command.json')==run.command_recipe(plan,root/'child',parent['child_pid']),'Actual full owned Docker launch recipe changed')
 control_bindings=[]
 if fresh_control_roots is not None:
  from validate_full_cache_shared_fresh40_v8 import complete_controls
  fresh_groups,control_bindings=complete_controls(root,fresh_control_roots)
 from full_cache_semantic_phase_stream_v1 import all_semantic_rows,FileLines,phase_range_from_snapshot,ack_binding,snapshot_binding
 all_events=all_semantic_rows(root/'child/api-native-trace.jsonl');all_trace=FileLines(root/'child/engine.combined.log');ctrl.require(len(child['phase_receipts'])==len(plan['schedule']),'Actual complete preregistered phase count missing');recollected=[];comparisons=[]
 from full_cache_shared_raw49_v2 import collect,compare49
 history=None;history_prior=None
 for index,(prototype,saved) in enumerate(zip(plan['schedule'],child['phase_receipts']),1):
  phase=prototype
  if prototype.get('dynamic_own_history_required'):
   from full_cache_shared_history_decode_v8 import finalized_binding as own_decode
   from full_cache_shared_history_v2 import canonical_sha,render_binding,resolve_history
   if history is None:history=own_decode(root,root/'child/own-history-decode')
   derivation=history['derivation'];request={'messages':derivation['messages'],'messages_sha256':canonical_sha(derivation['messages']),'template_kwargs':{'enable_thinking':False}};rendered=render_binding(request,saved['producer_ack']['render_observation'],derivation['engine_pid']);prime=next(r for r in child['phase_receipts'] if r['phase']['name']=='prime1');original_ids=prime['phase']['rows'][0]['ids']
   phase=resolve_history(prototype,derivation,rendered,original_ids,history_prior['inventory'] if history_prior else None,history_prior['rid'] if history_prior else None)
  from full_cache_shared_client_recollection_v8 import recollect as client_recollect
  path=root/'child/phases'/phase['name'];client_recollect(path/'client',saved['client'],[r['messages']for r in phase['rows']],phase['max_new'],phase['policy'],plan['research_alias']);ctrl.require(ctrl.read(path/'receipt.json')==saved and ctrl.read(path/'phase.recipe.json')==phase,'Actual original phase recipe/receipt differs');ack=saved['producer_ack'];marks=[e for e in all_events if e['kind']=='fullcache_phase_begin' and e['index']==index and e['name']==phase['name']];ctrl.require(len(marks)==1 and marks[0]['sequence']==ack['sequence'] and marks[0]['engine_pid']==ack['engine_pid'],'Actual producer phase marker/ack differs');events=phase_events(all_events,ack['sequence'],saved['end_sequence']);work=calls_and_work(phase,events);proof=ctrl.read(path/'waiter-proof.json');ctrl.require(proof==saved['waiter_proof']and proof['request']=={'index':index,'name':phase['name'],'calls':sorted(e['call']for e in events if e['kind']=='engine_begin')}and sum(e==proof['actual_event']for e in events)==1 and proof['actual_event']['kind']=='eos_waiter_phase_proof'and proof['actual_event']['request']==proof['request']and proof['actual_event']['proof']==proof['proof'],'Actual saved phase waiter proof/request/originalevent differs');from eos_waiter_terminal_source40_v2 import adjudicate
  actual_API=adjudicate(events,proof['proof']);actual_terminals={r['call']:r for r in actual_API['actual_API_terminal_associations']};ctrl.require(actual_API==saved['API_waiter_adjudication']==ctrl.read(path/'API-waiter-adjudication.json'),'Exact source waiter API adjudication differs');ctrl.require(work==saved['work'] and json.loads(json.dumps(actual_terminals))==json.loads(json.dumps(saved['terminal_associations'])),'Actual source counters/native terminal recollection differs');stream=ctrl.read(path/'semantic-stream.json');snapshot_binding(root/'child',stream,ack,ctrl.read(root/'child/memory-samples'/('phase-'+str(index)+'-before')/'sample.json'));trace,bounds=phase_range_from_snapshot(root/'child',stream);ctrl.require(bounds==saved['original_line_bounds'],'Actual original global source line indices changed')
  from full_cache_shared_capture_roster_v2 import observe_phase_rids
  from full_cache_shared_metadata_v2 import records as metadata_records,request_binding
  observe_phase_rids(phase,work)
  ctrl.require(ctrl.read(path/'producer-ack.json')==ack and ack_binding(ack,marks[0]),'Actual producer marker/render observation must equal its original ACK')
  observed=metadata_records(trace,ack['engine_pid'])
  main_body=[request_binding(observed,r['native_first_command']['line'],plan['scenario_binding']['selected_capture_pass']['selected_actual_RIDs']) for r in work]
  ctrl.require(main_body==saved['main_body_metadata'],'Actual command-local source40 main-body metadata changed')
  ctrl.require(native_lifetime(events,proof['proof'])==saved['native_lifetime_binding'],'Actual complete FC39/native protocol lifetime join changed')
  if phase.get('stale_owner_control_required'):
   from full_cache_shared_stale_owner_v2 import admit_proposal as stale_admit,recollect as stale_recollect
   control=ctrl.read(path/'stale-control.json');proposal=control['request_and_ack']['proposal'];sends=[e for e in events if e['kind']=='fullcache_control_send' and e['line']==proposal['command']];ctrl.require(len(sends)==1,'Exact actual original stale control send missing');stale_admit(proposal,[e for e in all_events if e['sequence']<sends[0]['sequence']]);terminals=actual_terminals;end=[t for t in terminals.values() if t['rid']==proposal['current_owner']['rid']];ctrl.require(len(end)==1,'Actual new owner normal terminal missing');binding=stale_recollect(proposal,sends[0],events,end[0]);ctrl.require(binding==saved['stale_owner_control']==control['binding'] and control['request_and_ack']['producer_ack']==ctrl.read(root/'child/control.ack.json') and control['request_and_ack']['request_packet']==ctrl.read(root/'child/control.request.json'),'Actual original stale request/ACK/source refusal/continuation binding differs')
  else:ctrl.require(saved['stale_owner_control'] is None,'Unexpected stale control claimed outside declared family')
  if phase['name']=='history_pinned':
   items=[r for r in observed if r['kind']=='checkpoint_inventory' and r['phase']=='main_body_complete' and r['rid']==work[0]['rid']];ctrl.require(len(items)==1,'Exact own earlier history checkpoint inventory required');history_prior={'inventory':items[0],'rid':work[0]['rid']}
  ctrl.require((saved['raw49'] is not None)==(phase['raw_capture_requested'] and phase['cancel_kind']!='prefill'),'Exact selected versus unselected phase raw coverage differs')
  if saved['raw49'] is not None:
   required=__import__('full_cache_shared_raw49_v2').required_roles(phase,work,actual_terminals);raw=collect(trace,events,work,[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],root/'child/captures',required);ctrl.require(json.loads(json.dumps(raw))==json.loads(json.dumps(saved['raw49'])),'Actual fresh full49 raw recollection changed');recollected.append(raw)
   if fresh_groups is not None:
    for group in raw['groups']:
     key=hashlib.sha256(b''.join(int(t).to_bytes(4,'little') for t in group['input_ids'])).hexdigest();ctrl.require(key in fresh_groups,'Actual genuine fresh49 control missing');comparisons.append(compare49(group,fresh_groups[key]))
 from full_cache_shared_persisted_v2 import fingerprint_refusal
 negative=ctrl.read(root/'child/parallel-persisted-negative.json');ctrl.require(negative['request_method']=='POST' and negative['request_url']=='http://127.0.0.1:'+str(plan['port'])+'/slots/0?action=save' and negative['request_body']=={'filename':'parallel-negative.bin'} and negative['real_endpoint_request'] is True,'Actual parallel persisted request association differs');ctrl.require(fingerprint_refusal(negative['status'],json.loads(negative['body']),True)==negative['source_refusal_binding'],'Actual original parallel persisted refusal differs')
 from full_cache_shared_reply_decode_v8 import finalized_binding as reply_decode_admit
 replies=reply_decode_admit(root,root/'child/own-reply-decode');ctrl.require(replies==child['own_reply_decode'],'Actual entire independently owned reply decode changed')
 final_waiter=ctrl.read(root/'child/waiter-final-proof.json');from eos_waiter_terminal_source40_v2 import adjudicate as all_waiter_adjudicate
 all_waiter_adjudicate(all_events,final_waiter)
 from full_cache_shared_launch_binding_v8 import binding as launch_binding
 launch_binding(root/'child',child,plan)
 from full_cache_shared_actor_retirement_v8 import saved_binding as retirement_binding
 retirement_binding(root/'child',child,ctrl.read(root/'child/launch.command.json')[ctrl.read(root/'child/launch.command.json').index('--name')+1],ctrl.read)
 from full_cache_shared_memory_v2 import account
 from full_cache_shared_metadata_v2 import records as memory_records
 pids={e['engine_pid'] for e in all_events if e['kind']=='engine_begin'};ctrl.require(len(pids)==1,'Actual resource accounting must remain in one native incarnation');samples=child['owned_host_memory_samples'];ctrl.require([(s['phase_index'],s['boundary']) for s in samples]==[(i,b) for i in range(1,len(plan['schedule'])+1) for b in ('before','after')],'Exact original phase host memory roster missing')
 from full_cache_shared_memory_capture_v8 import recollect as memory_recollect
 for sample in samples:memory_recollect(root/'child/memory-samples'/('phase-'+str(sample['phase_index'])+'-'+sample['boundary']),sample)
 resources=account(memory_records(all_trace,next(iter(pids))),samples);ctrl.require(resources==child['resource_accounting']==ctrl.read(root/'child/resource-accounting.json'),'Actual logical/expert/victim resource recollection changed')
 ctrl.require(ctrl.canonical(ctrl.manifest_binding(plan))==ctrl.canonical(before),'Current full shared source/SDK proof changed during reader');return {'current_parent_source_admitted':True,'actual_phase_count':len(child['phase_receipts']),'actual_completed_raw_phase_count':len(recollected),'fresh49_comparisons':comparisons,'all_supplied_comparisons_bitwise_equal':bool(comparisons) and all(r['all49_bitwise_equal'] for r in comparisons),'fresh_control_parent_and_source_binding_unobserved':fresh_control_roots is None,'actual_owned_fresh_control_bindings':control_bindings,'actual_recollected_resource_accounting':resources,'actual_own_API_yielded_ID_HTTP_text_decode':replies['derivation'],'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'latency_qualified':False,'remaining':full_scope_status({})}
