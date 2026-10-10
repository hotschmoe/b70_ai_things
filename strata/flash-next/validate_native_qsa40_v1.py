"""Strict original source40 model OFF/ON reader; own math remains diagnostic."""
from pathlib import Path
import native_qsa40_runtime_v1 as d
import native_qsa40_reader_v1 as raw
import native_qsa40_evidence_v1 as e
from serial37_canonical_json_v3 import canonical,matches_saved
from owned_layer3_qsa_evidence_v4 import artifacts
require=d.require

def finalized_binding(root):
 root=Path(root).resolve();report=d.read(root/'report.json');plan=d.read(root/'plan.snapshot.json');d.manifest(plan);require(type(report['schema'])is int and report['schema']==1 and report['passed']is True and report['errors']==[] and report['forced_cleanup']is False and report['captured_operands_used']is False and report['full_model_math_qualified']is False,'Exact actual source40 successful scoped parent required');require(report['plan_sha256']==d.sha(root/'plan.snapshot.json')and report['source_plan_sha256']==d.closure()and d.read(root/'source-plan.snapshot.json')==d.read(d.PLAN)and(root/'parent.snapshot.py').read_bytes()==Path(__file__).with_name('qualify_native_qsa40_v1.py').read_bytes(),'Exact original source/plan/parent snapshots differ');artifacts(root,report);require(report['cards']==plan['cards']and set(report['phases'])=={'off','on'},'Exact actual arm/card roster required');before=report['started_epoch']
 for label in('off','on'):
  ph=report['phases'][label];e.phase(root,label,plan,ph,report['started_epoch']);require(ph['ready']['parent']['pid']==report['producer_pid']and before<=ph['child_started_epoch'],'Actual sequential owned parent/child order differs');expected=[report['producer_interpreter'],str(d.HERE/'native_qsa40_runtime_v1.py'),'child','--plan',str(root/'plan.snapshot.json'),'--output',str(root/label)]+(['--on']if label=='on'else[]);require(ph['command']==expected and d.sha(report['producer_interpreter'])==report['producer_interpreter_sha256'],'Actual historical interpreter/full child CLI differs');before=ph['child_terminal_epoch']
 e.finite([report['model_computation_terminal_epoch'],report['GPU_terminal_epoch'],report['finished_epoch']]);require(before<=report['model_computation_terminal_epoch']<=report['GPU_terminal_epoch'],'Actual reference computation cannot outrun final CPU/GPU terminal');e.health(root,'post',report['post_health'],report['GPU_terminal_epoch']);e.journal(root,'post',report['post_journal'],report['started_epoch'],report['post_health']['finished_epoch'])
 from native_rms_publisher_binding_v2 import publisher_binding
 from native_rms_device_ops_lifecycle_v8 import pre_publisher
 lock=d.read(d.HERE/'model-lock.json');view=dict(report);view['pre_health']=report['phases']['off']['pre_health'];pre_publisher(root,view,lock,d.sha(d.HERE/'model-lock.json'),d.ROOT);require(matches_saved(report['post_full4'],root/'post-full4.json'),'Actual original full4 differs');publisher_binding(root,report,lock,d.sha(d.HERE/'model-lock.json'),d.ROOT)
 off=raw.arm(root/'off',plan,report['phases']['off'],False,report['plan_sha256']);on=raw.arm(root/'on',plan,report['phases']['on'],True,report['plan_sha256']);comparison=raw.comparisons(off,on);require(canonical(comparison)==canonical(report['comparison'])and matches_saved(comparison,root/'numerical-comparison.json'),'Actual complete full49/P30 numerical joins changed')
 from native_qsa40_input33_v1 import verify
 identity=d.read(Path(plan['own_producer_root'])/'report.json')['original_work_config']['model_identity'];original={'off':verify(off['source33'],identity),'on':verify(on['source33'],identity)};require(canonical(original)==canonical(report['original_input33'])and matches_saved(original,root/'original-input33.json'),'Actual original staged input33 comparison differs')
 from native_qsa40_localization_v1 import join
 localization=join(plan['own_producer_root'],on['targets']);require(canonical(localization)==canonical(report['localization'])and matches_saved(localization,root/'localization.json'),'Original independent vs actual native localization differs');d.manifest(plan);artifacts(root,report);require(d.read(root/'report.json')==report,'Actual parent changed during final raw admissions')
 return {'report_sha256':d.sha(root/'report.json'),'cards':plan['cards'],'engine_receipt_sha256':plan['candidate_binding']['prepared']['engine_receipt_sha256'],'actual_source40_observer_equivalence':True,'full49_pairs':196,'wholeprefix_logical_row_pairs':2304,'original_vs_native_localization_observed':True,'full_model_math_qualified':False,'runtime_graph_handle_association_observed':False}
