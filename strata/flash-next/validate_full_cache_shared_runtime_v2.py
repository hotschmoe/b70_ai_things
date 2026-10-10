"""Read-only closed phase/source admission; absent controls never become PASS."""
import hashlib,json,math
from pathlib import Path
import full_cache_shared_runtime_v2 as ctrl
import run_full_cache_shared_runtime_v2 as run
from full_cache_shared_phase_contract_v2 import calls_and_work,phase_events,full_scope_status
from full_cache_shared_terminal_v2 import associate_events
from full_cache_shared_native_lifetime_v2 import recollect as native_lifetime
from api_journal_exact_admission_v5 import journal_gate
from validate_batch_api_cache_positive_buffered_v5 import exact_health_gate
from source_page_watchdog_v3 import KNOWN_PAGES
def finalized_binding(directory,fresh_groups=None,fresh_control_roots=None):
 ctrl.require(fresh_groups is None,'Caller-supplied fresh arrays are not actual owned/source-qualified fresh49 controls; future explicit control admission required')
 root=Path(directory).resolve();parent=ctrl.read(root/'parent-qualification.json');plan=ctrl.read(root/'input-plan.snapshot.json');child=ctrl.read(root/'child/report.json');before=ctrl.manifest_binding(plan)
 ctrl.require(parent['passed'] is True and not parent['errors'] and parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and parent['interrupted'] is False and ctrl.canonical(parent['prepared_chain'])==ctrl.canonical(before),'Actual owned parent/source/health scope incomplete')
 ctrl.require(parent['wrapper_sha256']==ctrl.sha(Path(__file__).with_name('qualify_full_cache_shared_runtime_v2.py')) and parent['controller_sha256']==ctrl.sha(Path(ctrl.__file__)) and parent['child_report_sha256']==ctrl.sha(root/'child/report.json'),'Actual original producer/source/report association differs')
 external=Path(parent['plan']);snapshot=root/'child/plan.snapshot.json';ctrl.require(ctrl.sha(external)==parent['plan_sha256']==child['plan_sha256']==ctrl.sha(snapshot)==ctrl.sha(root/'input-plan.snapshot.json') and ctrl.read(external)==ctrl.read(snapshot)==plan,'Exact actual external/input/child plan bytes differ')
 cli=ctrl.read(root/'child.command.json');ctrl.require(cli==[child['producer_interpreter'],str(Path(ctrl.__file__)),'run','--plan',str(external.resolve()),'--pre-health',str(root/'pre-health.json'),'--output',str(root/'child')] and child['producer_pid']==parent['child_pid'] and ctrl.sha(child['producer_interpreter'])==child['producer_interpreter_sha256'],'Actual producer CLI/PID/interpreter differs')
 ctrl.require(child['collection_and_teardown_passed'] is True and child['error'] is None and child['removed'] is True and child['state']['ExitCode']==0 and not child['state']['Running'] and not child['state']['OOMKilled'],'Actual complete normal actor lifecycle absent');ctrl.require(ctrl.read(root/'child/launch.command.json')==run.command_recipe(plan,root/'child',parent['child_pid']),'Actual full owned Docker launch recipe changed')
 for name,w in child['artifact_bindings'].items():
  path=root/'child'/name;ctrl.require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to((root/'child').resolve()) and not path.is_symlink() and path.stat().st_size==w['bytes'] and ctrl.sha(path)==w['sha256'],'Actual sealed runtime artifact extent/SHA/path differs')
 for stage in ('pre','post'):exact_health_gate(root,stage)
 journal_gate(root,parent)
 lock_path=ctrl.HERE/'model-lock.json';lock=ctrl.read(lock_path);identity_path=root/'post-model-identity.json';identity=ctrl.read(identity_path);ctrl.require(parent['post_model_identity']['path']==str(identity_path) and parent['post_model_identity']['sha256']==ctrl.sha(identity_path) and identity['passed'] is True and len(identity['rows'])==4 and identity['lock_sha256']==ctrl.sha(lock_path) and identity['model_revision']==lock['revision'],'Actual current new4 source receipt changed')
 expected={str(ctrl.ROOT/lock['destination']/r['path']):r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')};ctrl.require({r['path'] for r in identity['rows']}==set(expected),'Actual publisher source roster differs')
 for row in identity['rows']:
  want=expected[row['path']];ctrl.require(row['passed'] is True and row['sha256']==row['expected_sha256']==want['sha256'] and row['bytes']==want['size'] and row['stat_before']==row['stat_after']==ctrl.c1.stat_signature(Path(row['path'])),'Actual current source publisher/hash/stat differs')
 third=next(r for r in identity['rows'] if '00003-of-00004' in r['path'])
 for label in ('known_pages_before_hash','known_pages_after_hash'):
  page=parent[label];ctrl.require(page['passed'] is True and page['path']==third['path'] and page['stat_before']==page['stat_after']==third['stat_after'] and [(r['offset'],r['expected_sha256']) for r in page['rows']]==list(KNOWN_PAGES) and all(r['bytes']==4096 and r['sha256']==r['expected_sha256'] and r['passed'] is True for r in page['rows']),'Both exact current source page views required')
 ctrl.require(parent['known_pages_before_hash']['epoch']<=identity['started']<=identity['finished']<=parent['known_pages_after_hash']['epoch']<=parent['finished_epoch'],'Current page/full4/parent chronology differs')
 control_bindings=[]
 if fresh_control_roots is not None:
  from validate_full_cache_shared_fresh39_v2 import complete_controls
  fresh_groups,control_bindings=complete_controls(root,fresh_control_roots)
 all_events=run.events(root/'child/api-native-trace.jsonl');all_trace=(root/'child/engine.combined.log').read_text();ctrl.require(len(child['phase_receipts'])==len(plan['schedule']),'Actual complete preregistered phase count missing');recollected=[];comparisons=[]
 from full_cache_shared_raw49_v2 import collect,compare49
 history=None;history_prior=None
 for index,(prototype,saved) in enumerate(zip(plan['schedule'],child['phase_receipts']),1):
  phase=prototype
  if prototype.get('dynamic_own_history_required'):
   from full_cache_shared_history_decode_v2 import finalized_binding as own_decode
   from full_cache_shared_history_v2 import canonical_sha,render_binding,resolve_history
   if history is None:history=own_decode(root,root/'child/own-history-decode')
   derivation=history['derivation'];request={'messages':derivation['messages'],'messages_sha256':canonical_sha(derivation['messages']),'template_kwargs':{'enable_thinking':False}};rendered=render_binding(request,saved['producer_ack']['render_observation'],derivation['engine_pid']);prime=next(r for r in child['phase_receipts'] if r['phase']['name']=='prime1');original_ids=prime['phase']['rows'][0]['ids']
   phase=resolve_history(prototype,derivation,rendered,original_ids,history_prior['inventory'] if history_prior else None,history_prior['rid'] if history_prior else None)
  path=root/'child/phases'/phase['name'];ctrl.require(ctrl.read(path/'receipt.json')==saved and ctrl.read(path/'phase.recipe.json')==phase,'Actual original phase recipe/receipt differs');ack=saved['producer_ack'];marks=[e for e in all_events if e['kind']=='fullcache_phase_begin' and e['index']==index and e['name']==phase['name']];ctrl.require(len(marks)==1 and marks[0]['sequence']==ack['sequence'] and marks[0]['engine_pid']==ack['engine_pid'],'Actual producer phase marker/ack differs');events=phase_events(all_events,ack['sequence'],saved['end_sequence']);work=calls_and_work(phase,events);ctrl.require(work==saved['work'] and json.loads(json.dumps(associate_events(events,{r['call'] for r in work})))==json.loads(json.dumps(saved['terminal_associations'])),'Actual source counters/native terminal recollection differs');trace,bounds=run.phase_trace(all_trace,phase['name'],index);ctrl.require(bounds==saved['original_line_bounds'],'Actual original global source line indices changed')
  from full_cache_shared_capture_roster_v2 import observe_phase_rids
  from full_cache_shared_metadata_v2 import records as metadata_records,request_binding
  observe_phase_rids(phase,work)
  ctrl.require(ctrl.read(path/'producer-ack.json')==ack and {k:marks[0][k] for k in ack}==ack,'Actual producer marker/render observation must equal its original ACK')
  observed=metadata_records(trace,ack['engine_pid'])
  main_body=[request_binding(observed,r['native_first_command']['line'],plan['scenario_binding']['selected_capture_pass']['selected_actual_RIDs']) for r in work]
  ctrl.require(main_body==saved['main_body_metadata'],'Actual command-local source39 main-body metadata changed')
  ctrl.require(native_lifetime(events)==saved['native_lifetime_binding'],'Actual complete FC39/native protocol lifetime join changed')
  if phase.get('stale_owner_control_required'):
   from full_cache_shared_stale_owner_v2 import admit_proposal as stale_admit,recollect as stale_recollect
   control=ctrl.read(path/'stale-control.json');proposal=control['request_and_ack']['proposal'];sends=[e for e in events if e['kind']=='fullcache_control_send' and e['line']==proposal['command']];ctrl.require(len(sends)==1,'Exact actual original stale control send missing');stale_admit(proposal,[e for e in all_events if e['sequence']<sends[0]['sequence']]);terminals=associate_events(events,{r['call'] for r in work});end=[t for t in terminals.values() if t['rid']==proposal['current_owner']['rid']];ctrl.require(len(end)==1,'Actual new owner normal terminal missing');binding=stale_recollect(proposal,sends[0],events,end[0]);ctrl.require(binding==saved['stale_owner_control']==control['binding'] and control['request_and_ack']['producer_ack']==ctrl.read(root/'child/control.ack.json') and control['request_and_ack']['request_packet']==ctrl.read(root/'child/control.request.json'),'Actual original stale request/ACK/source refusal/continuation binding differs')
  else:ctrl.require(saved['stale_owner_control'] is None,'Unexpected stale control claimed outside declared family')
  if phase['name']=='history_pinned':
   items=[r for r in observed if r['kind']=='checkpoint_inventory' and r['phase']=='main_body_complete' and r['rid']==work[0]['rid']];ctrl.require(len(items)==1,'Exact own earlier history checkpoint inventory required');history_prior={'inventory':items[0],'rid':work[0]['rid']}
  ctrl.require((saved['raw49'] is not None)==(phase['raw_capture_requested'] and phase['cancel_kind']!='prefill'),'Exact selected versus unselected phase raw coverage differs')
  if saved['raw49'] is not None:
   required=__import__('full_cache_shared_raw49_v2').required_roles(phase,work,associate_events(events,{r['call'] for r in work}));raw=collect(trace,events,work,[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],root/'child/captures',required);ctrl.require(json.loads(json.dumps(raw))==json.loads(json.dumps(saved['raw49'])),'Actual fresh full49 raw recollection changed');recollected.append(raw)
   if fresh_groups is not None:
    for group in raw['groups']:
     key=hashlib.sha256(b''.join(int(t).to_bytes(4,'little') for t in group['input_ids'])).hexdigest();ctrl.require(key in fresh_groups,'Actual genuine fresh49 control missing');comparisons.append(compare49(group,fresh_groups[key]))
 from full_cache_shared_persisted_v2 import fingerprint_refusal
 negative=ctrl.read(root/'child/parallel-persisted-negative.json');ctrl.require(negative['request_method']=='POST' and negative['request_url']=='http://127.0.0.1:'+str(plan['port'])+'/slots/0?action=save' and negative['request_body']=={'filename':'parallel-negative.bin'} and negative['real_endpoint_request'] is True,'Actual parallel persisted request association differs');ctrl.require(fingerprint_refusal(negative['status'],json.loads(negative['body']),True)==negative['source_refusal_binding'],'Actual original parallel persisted refusal differs')
 from full_cache_shared_launch_binding_v2 import binding as launch_binding
 launch_binding(root/'child',child,plan)
 from full_cache_shared_memory_v2 import account
 from full_cache_shared_metadata_v2 import records as memory_records
 pids={e['engine_pid'] for e in all_events if e['kind']=='engine_begin'};ctrl.require(len(pids)==1,'Actual resource accounting must remain in one native incarnation');samples=child['owned_host_memory_samples'];ctrl.require([(s['phase_index'],s['boundary']) for s in samples]==[(i,b) for i in range(1,len(plan['schedule'])+1) for b in ('before','after')],'Exact original phase host memory roster missing')
 from full_cache_shared_memory_capture_v2 import recollect as memory_recollect
 for sample in samples:memory_recollect(root/'child/memory-samples'/('phase-'+str(sample['phase_index'])+'-'+sample['boundary']),sample)
 resources=account(memory_records(all_trace,next(iter(pids))),samples);ctrl.require(resources==child['resource_accounting']==ctrl.read(root/'child/resource-accounting.json'),'Actual logical/expert/victim resource recollection changed')
 ctrl.require(ctrl.canonical(ctrl.manifest_binding(plan))==ctrl.canonical(before),'Current full shared source/SDK proof changed during reader');return {'current_parent_source_admitted':True,'actual_phase_count':len(child['phase_receipts']),'actual_completed_raw_phase_count':len(recollected),'fresh49_comparisons':comparisons,'all_supplied_comparisons_bitwise_equal':bool(comparisons) and all(r['all49_bitwise_equal'] for r in comparisons),'fresh_control_parent_and_source_binding_unobserved':fresh_control_roots is None,'actual_owned_fresh_control_bindings':control_bindings,'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'latency_qualified':False,'remaining':full_scope_status({})}
