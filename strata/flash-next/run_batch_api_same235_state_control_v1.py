"""NEW exact-target serial control on pinned V6 owned lifecycle, originals untouched."""
from pathlib import Path
import run_batch_api_cache_positive_buffered_v6 as original
import same235_state_control_v1 as controls
import same235_state_control_protocol_v1 as protocol
import batch_numerical_proofs_v7 as proofs
command_recipe=original.command_recipe

def run(plan,out,pre_health,diagnostic=True):
 source=Path(original.__file__).read_text();start=source.index('  def allow_cancel():');end=source.index('\n except BaseException as exc:',start)
 replacement="""  result['serial_state_control_client']=controls.collect_controls(url,'hotschmoe-dd',plan['positive_V6_source_preparation'],out/'state-controls',trace_path=out/'api-native-trace.jsonl')
  semantic=protocol.events(out/'api-native-trace.jsonl');result['actual_serial_state_controls']=protocol.recollect(semantic,result['serial_state_control_client']['rows'],len(plan['stage_ranges']))
  result.update(initial_native_and_api_collection_complete=True,diagnostic=diagnostic,actual_cached_state_handoff_qualified=False)
"""
 source=source[:start]+replacement+source[end:];scope=source[:source.index("if __name__=='__main__':")];ns=dict(vars(original));ns.update(__file__=__file__,__name__='same235_owned_control_runtime_v1',controls=controls,protocol=protocol,jsonlines=protocol.events,seal_report=seal_report);exec(compile(scope,str(original.__file__)+'[NEW-serial-state-control]','exec'),ns)
 # The copied source defines seal_report; use only NEW explicit control sealing.
 ns['seal_report']=seal_report;ns['jsonlines']=protocol.events;return ns['run'](plan,out,pre_health,diagnostic)

def seal_report(plan,out,result):
 original.require(original.read(out/'plan.snapshot.json')==plan,'Actual same235 producer plan snapshot differs');original.seal_report(plan,out,result);result.update(same235_state_control_generation=1,buffered_wrapper_generation=7,underlying_EOS_cause_established=False,matched_API_concurrent_fresh_control_still_required=True,actual_cached_state_handoff_qualified=False,full_model_math_qualified=False);result['artifact_bindings']=proofs.artifact_bindings(out);original.write(out/'report.json',result)
 return result
