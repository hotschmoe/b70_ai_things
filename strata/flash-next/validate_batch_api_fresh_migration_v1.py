"""Readonly actual capability-on/fresh recompute collection; no cached handoff claim."""
import json
from pathlib import Path
import batch_api_fresh_migration_v1 as ctrl
from api_fresh_migration_contract_v1 import actual_startup,actual_fresh_legs,actual_resets
from audit_batch_numerical_suite_v7 import final_source_join,vectors
from batch_numerical_proofs_v7 import validate_artifacts,raw_case,terminal_api_join,owner_proofs
import qualify_batch_numerical_v7 as shared_parent

def exact_health_gate(root,stage):
 h=ctrl.read(root/(stage+'-health.json'));ctrl.require(h['passed'] is True and h['cards']==[0,1] and h['health_image']==shared_parent.HEALTH and len(h['files'])==2,'Actual strict percard/pair health absent')
 commands=[[str(ctrl.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',shared_parent.HEALTH],[str(ctrl.ROOT/'bin/xpu-collective-health'),'--img',shared_parent.HEALTH,'--p2p','0','--timeout','180']]
 for row,command,label in zip(h['files'],commands,('strict','compiled-pair')):
  path=root/(stage+'-'+label+'.log');command_path=root/(stage+'-'+label+'.command.json')
  ctrl.require(Path(row['path']).resolve()==path.resolve() and row['return_code']==0 and row['error'] is None and ctrl.sha(path)==row['sha256'] and row['command']==command==ctrl.read(command_path) and ctrl.sha(command_path)==row['command_file_sha256'],'Actual health exact command/path/log association differs')
 return h

def finalized_binding(run_root):
 root=Path(run_root).resolve();parent=ctrl.read(root/'parent-qualification.json');plan=ctrl.read(root/'input-plan.snapshot.json');child=ctrl.read(root/'child/report.json');chain=ctrl.manifest_binding(plan)
 ctrl.require(parent['passed'] is True and parent['child_return_code']==0 and parent['owned_containers_terminal'] is True and parent['forced_cleanup'] is False and parent['interrupted'] is False and not parent['errors'] and all(parent[k] is True for k in ('pre_health_passed','post_health_passed','kernel_fault_gate_passed')),'Actual API parent source/health/owned gates failed')
 ctrl.require(parent['scoped_collection_or_serial_arm_qualified'] is True and parent['source_guard_generation']==3,'Actual qualified collection/source-page watchdog generation absent')
 ctrl.require(parent['wrapper_sha256']==ctrl.sha(Path(__file__).with_name('qualify_batch_api_fresh_migration_v1.py')) and parent['controller_sha256']==ctrl.sha(Path(ctrl.__file__)) and parent['prepared_chain']==chain and parent['child_report_sha256']==ctrl.sha(root/'child/report.json'),'Actual API producer/parent/source association differs')
 ctrl.require((root/'controller.py').read_bytes()==Path(ctrl.__file__).read_bytes() and (root/'wrapper.py').read_bytes()==Path(__file__).with_name('qualify_batch_api_fresh_migration_v1.py').read_bytes(),'Actual producer source snapshots differ')
 cli=ctrl.read(root/'child.command.json');ctrl.require(Path(cli[0]).resolve()==Path(child['producer_interpreter']).resolve() and ctrl.sha(Path(child['producer_interpreter']))==child['producer_interpreter_sha256'] and cli[1:]==[str(Path(ctrl.__file__)), 'run','--plan',str(Path(parent['plan']).resolve()),'--pre-health',str(root/'pre-health.json'),'--output',str(root/'child')] and child['producer_pid']==parent['child_pid'],'Actual child CLI/interpreter/PID association differs')
 ctrl.require(child['schema']==5 and child['harness_generation']==8 and child['api_fresh_recompute_generation']==1 and child['collection_and_teardown_passed'] is True and child['actual_cached_state_handoff_qualified'] is False and child['full_cache_public_fresh_pin_qualified'] is False and child['full_model_math_qualified'] is False,'Actual API scope/collection differs');validate_artifacts(root/'child',child['artifact_bindings']);source=final_source_join(root,parent,plan,child)
 ctrl.require(ctrl.read(root/'child/launch.command.json')==ctrl.api.command_recipe(plan,root/'child',parent['child_pid']),'Actual API full owned launch command differs')
 cfg=ctrl.read(Path(plan['prepared'])/'server-config.json');cfg.update(args=plan['args'],env=plan['env'],port=plan['port'],parallel=plan['slots'],log='/results/server-engine.log',model_name='hotschmoe-dd',aliases=[plan['research_alias']]);ctrl.require(cfg==ctrl.read(root/'child/server-config.json'),'Actual full API configuration differs')
 models=ctrl.read(root/'child/models.json');identity=ctrl.sha(root/'child/artifact-identity.json');ctrl.require([r['id'] for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==identity for r in models['data']),'Actual served identity differs')
 for stage in ('pre','post'):exact_health_gate(root,stage)
 state=child['state'];ctrl.require(child['actual_native_exit_zero'] is True and child['removed'] is True and state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'] and child['error'] is None,'Actual API terminal/owned cleanup failed')
 events=ctrl.api.jsonlines(root/'child/api-native-trace.jsonl');begins={e['call']:e for e in events if e['kind']=='engine_begin'};ends={e['call']:e for e in events if e['kind']=='engine_end'};ctrl.require(len(begins)==2*plan['slots'] and set(begins)==set(ends),'Actual full warm/target request roster missing')
 fresh=actual_fresh_legs(events,set(begins));ctrl.require(fresh==child['actual_all_fresh_legs'],'Actual native fresh send proof differs');trace=(root/'child/engine.combined.log').read_text();ctrl.require(actual_resets(trace,events,set(begins),len(plan['stage_ranges']))==child['actual_all_stage_resets'],'Actual source all-stage reset records differ');startup=actual_startup(trace,plan['slots']);ctrl.require(startup==child['actual_native_startup'],'Actual geometry startup differs')
 target={};logical={}
 for index,messages in enumerate(plan['messages']['target']):
  calls=[call for call,e in begins.items() if e['rendered_prompt']['messages']==messages];ctrl.require(len(calls)==1 and begins[calls[0]]['submitted_ids']==plan['api_token_ids']['target'][index],'Actual target input/tokenizer/call association differs');target[calls[0]]=ends[calls[0]];logical[index]=ends[calls[0]]['rid']
 cancelled={call for call,e in target.items() if e['cancelled']};ctrl.require(len(cancelled)==1,'Actual single target cancellation required');terminal_api_join([e for e in events if e.get('call') in target or e['kind']=='native_receive'],set(target),cancelled)
 stages=[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])];jobs=ctrl.read(root/'child/serial-jobs.json')['jobs'];prefixes=ctrl.api.prefix_jobs(events);ctrl.require(prefixes==ctrl.read(root/'child/prefixes.json') and jobs==[j for j in prefixes['jobs'] if j['call'] in target],'Actual consumed-prefix jobs/native history differs');migrating=sorted({j['rid'] for j in jobs if j['role']=='solo_migration'});policy={(logical[r['logical_index']],r['role']):r['values'] for r in plan['actual_counter_policy']};recollected=raw_case(trace,events,list(logical.values()),migrating,stages,root/'child/captures',policy,ownership_terminal=False) if plan['diagnostic'] else None
 if plan['diagnostic']:ctrl.require(json.loads(json.dumps(recollected))==child['raw'],'Actual full raw API fidelity recollection differs');vectors(root)
 ownership=owner_proofs(trace,stages,plan['slots'],plan['lane'],snapshots_observed=bool(plan['diagnostic']));ctrl.require(json.loads(json.dumps(ownership))==child['logical_owner_proofs'],'Actual terminal owner recollection differs')
 source_after=final_source_join(root,parent,plan,child);ctrl.require(ctrl.manifest_binding(plan)==chain,'Actual current API source proof changed during readonly recollection')
 return parent,child,plan,{'actual_fresh_legs':fresh,'actual_geometry_startup':startup,'source_before':source,'source_after':source_after,'actual_cached_state_handoff_qualified':False,'serial_numerical_comparison_qualified':False,'latency_qualified':False,'full_model_math_qualified':False,'remaining_positive_cached_handoff_required':True}
