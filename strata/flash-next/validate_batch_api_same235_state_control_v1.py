"""Readonly actual positive cached-slot transfer collection; no cached handoff claim."""
import json
from pathlib import Path
import batch_api_same235_state_control_v1 as ctrl
import same235_state_control_protocol_v1 as protocol
from api_shortwarm_positive_contract_v6 import actual_startup,actual_phase_legs,last_live_handoff
from api_owned_terminal_association_v2 import associate_events
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
 ctrl.require(parent['wrapper_sha256']==ctrl.sha(Path(__file__).with_name('qualify_batch_api_same235_state_control_v1.py')) and parent['controller_sha256']==ctrl.sha(Path(ctrl.__file__)) and parent['prepared_chain']==chain and parent['child_report_sha256']==ctrl.sha(root/'child/report.json'),'Actual API producer/parent/source association differs')
 ctrl.require((root/'controller.py').read_bytes()==Path(ctrl.__file__).read_bytes() and (root/'wrapper.py').read_bytes()==Path(__file__).with_name('qualify_batch_api_same235_state_control_v1.py').read_bytes(),'Actual producer source snapshots differ')
 cli=ctrl.read(root/'child.command.json');ctrl.require(Path(cli[0]).resolve()==Path(child['producer_interpreter']).resolve() and ctrl.sha(Path(child['producer_interpreter']))==child['producer_interpreter_sha256'] and cli[1:]==[str(Path(ctrl.__file__)), 'run','--plan',str(Path(parent['plan']).resolve()),'--pre-health',str(root/'pre-health.json'),'--output',str(root/'child')] and child['producer_pid']==parent['child_pid'],'Actual child CLI/interpreter/PID association differs')
 ctrl.require(child['schema']==6 and child['harness_generation']==9 and child['api_cache_positive_generation']==2 and child['buffered_trace_generation']==3 and child['terminal_association_generation']==2 and child['buffered_wrapper_generation']==7 and child['collection_and_teardown_passed'] is True and child['actual_cached_state_handoff_qualified'] is False and child['full_cache_public_fresh_pin_qualified'] is False and child['full_model_math_qualified'] is False,'Actual API scope/collection differs');validate_artifacts(root/'child',child['artifact_bindings']);ctrl.require(child['same235_state_control_generation']==1 and child['underlying_EOS_cause_established'] is False and child['shortwarm_case_generation']==1 and child['matched_buffer_only_A_B_input_equivalent'] is False,'Separate shortwarm scope required');source=final_source_join(root,parent,plan,child)
 ctrl.require(ctrl.read(root/'child/launch.command.json')==ctrl.api.command_recipe(plan,root/'child',parent['child_pid']),'Actual API full owned launch command differs')
 cfg=ctrl.read(Path(plan['prepared'])/'server-config.json');cfg.update(args=plan['args'],env=plan['env'],port=plan['port'],parallel=plan['slots'],log='/results/server-engine.log',model_name='hotschmoe-dd',aliases=[plan['research_alias']]);ctrl.require(cfg==ctrl.read(root/'child/server-config.json'),'Actual full API configuration differs')
 models=ctrl.read(root/'child/models.json');identity=ctrl.sha(root/'child/artifact-identity.json');ctrl.require([r['id'] for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==identity for r in models['data']),'Actual served identity differs')
 for stage in ('pre','post'):exact_health_gate(root,stage)
 from api_journal_exact_admission_v5 import journal_gate
 journal_gate(root,parent)
 status=ctrl.read(root/'child/api-native-trace.jsonl.buffered-status.json');ctrl.require(status==child['buffered_trace_status'] and status['passed'] is True and status['closed'] is True and status['periodic_thread_retired'] is True and status['ARM_thread_retired'] is True and status['observer_source_generation']==3,'Actual buffered drain/status binding differs')
 state=child['state'];ctrl.require(child['actual_native_exit_zero'] is True and child['removed'] is True and state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'] and child['error'] is None,'Actual API terminal/owned cleanup failed')
 events=protocol.events(root/'child/api-native-trace.jsonl');controls=ctrl.read(root/'child/state-controls/client-controls.json');ctrl.require(controls==child['serial_state_control_client'] and controls['positive_configuration_origin_plan']==plan['positive_V6_source_preparation'],'Actual same235 controls/source origin differ')
 recollected=protocol.recollect(events,controls['rows'],len(plan['stage_ranges']));ctrl.require(recollected==child['actual_serial_state_controls'] and [row['declared_job'] for row in controls['rows']]==plan['serial_state_control_jobs'],'Actual state control raw/protocol/input recollection differs')
 for row in controls['rows']:
  directory=root/'child/state-controls'/row['declared_job']['control']
  for phase,key in [('warm','actual_warm_client'),('target','actual_target_client')]:
   actual=ctrl.read(directory/phase/'client.json');ctrl.require(actual==row[key] and actual['client_transport_completed'] is True,'Actual control client records differ');models=ctrl.read(directory/phase/'models.json');ctrl.require([r['id'] for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==identity for r in models['data']),'Actual control request served identity differs')
 source_after=final_source_join(root,parent,plan,child);ctrl.require(ctrl.manifest_binding(plan)==chain,'Current state control source changed during recollection')
 return parent,child,plan,{'actual_serial_state_controls':recollected,'source_before':source,'source_after':source_after,'same235_state_control_generation':1,'original_failed_V6_accepted_as_success':False,'underlying_EOS_cause_established':False,'matched_API_concurrent_fresh_control_still_required':True,'actual_cached_state_handoff_qualified':False,'full_model_math_qualified':False,'latency_qualified':False}
