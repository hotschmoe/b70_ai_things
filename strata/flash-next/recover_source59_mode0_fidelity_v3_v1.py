"""Read-only component recovery of exact failed V3 parent; no status/file rewrite."""
import argparse,ast,copy,json,tempfile
from pathlib import Path
from types import FunctionType,SimpleNamespace
import validate_source59_mode0_fidelity_v3 as original
import source59_mode0_fidelity_raw_v3 as raw_original
import source59_literal1_numeric_lpdisabled_v1 as numeric
from serial37_canonical_json_v3 import canonical
c=original.c;require=original.require;actor=original.actor;HERE=actor.HERE;ROOT=actor.ROOT
PLAN=HERE/'source59-mode0-fidelity-recovery-source-plan-v1.json'
REPORT_SHA='2efcaa63076d6421ea3476e29aea62c44e38809d2199c42ca23d6a54ae947bc8'
ERROR='ValueError: Actual exact fresh4/MAX1/absentPIN request required'
def source_binding():
 source=c.sha(PLAN)
 for n,h in c.read(PLAN)['files'].items():require(c.sha(ROOT/n)==h,'Current59 recovery source changed '+n)
 require(c.sha(PLAN)==source,'Recovery plan changed');return source

def phase_ports(active):
 ports=original.boundary_reader_ports(active)
 # Reuse every original phase command/ownership/EOF/health/pipe/startup/launch
 # predicate. The failed post-native collector left report.results empty, so
 # explicitly join authentic result.json/phase_receipt rather than invent it.
 import validate_source41_model_cli_v5 as grammar
 node=copy.deepcopy(next(n for n in ast.parse(Path(grammar.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='phase_binding'));count=0
 for item in ast.walk(node):
  if isinstance(item,ast.Call)and isinstance(item.func,ast.Name)and item.func.id=='canonical'and item.args and ast.unparse(item.args[0])=="report['results'][phase]":item.args[0]=ast.Name(id='result',ctx=ast.Load());count+=1
 require(count==1,'Exactly original missing report-result association required')
 ports['original_result_receipt_join']=True
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[failed-postcollector-missing-parent-row-only]','exec'),ports)
 return ports

def collect_actual(directory,binding):
 namespace=dict(raw_original.collect.__globals__)
 namespace['numeric_software']=SimpleNamespace(port=lambda:(numeric.port(),None,None))
 node=copy.deepcopy(next(n for n in ast.parse(Path(raw_original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='collect'));count=0
 for item in ast.walk(node):
  if isinstance(item,ast.ImportFrom)and item.module=='source41_model_cli_contract_v1'and [x.name for x in item.names]==['consume']:item.module='native_qsa41_targets_v1';count+=1
 require(count==1,'Exactly one missing raw-consume import route required')
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[existing-raw-consumer-import-only]','exec'),namespace)
 return namespace['collect'](directory,binding)

def recovery_gate(report,digest):
 require(digest==REPORT_SHA and report['passed']is False and report['errors']==[ERROR]and report['results']=={}and report['SDK_receipt_sha256']==actor.SDK_SHA and report['source_plan_sha256']==c.sha(actor.PLAN),'Only authentic failedV3 post-native formatter attempt allowed')
 require(report['post_full4']['passed']is True,'Failed parent must still have completed publisher close')

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();digest=c.sha(root/'report.json');report=c.read(root/'report.json');recovery_gate(report,digest);roster=original.tree(root);require(roster==report['artifacts'],'Original failed producer artifact roster differs');phase=actor.PHASES[0];directory=root/phase;recipe=c.read(directory/'recipe.json')
 try:
  with original.configured(report,recipe)as active:
   require(actor.closure()==report['source_plan_sha256'] and c.sha(root/'source-plan.snapshot.json')==report['source_plan_sha256']and c.sha(root/'producer.snapshot.py')==c.sha(actor.__file__),'Exact actual V3 producer/source snapshot required')
   plan=c.read(root/'plan.snapshot.json');require(c.sha(root/'plan.snapshot.json')==report['plan_sha256'],'Original corpus plan changed');m=c.read(root/'prepared.snapshot.json');cfg=c.read(root/'config.snapshot.json');inputs=c.read(root/'inputs.json')
   require(c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256'] and c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256']and canonical(c.read(Path(plan['template_path'])/'prepared.json'))==canonical(m)and canonical(c.read(Path(plan['template_path'])/'server-config.json'))==canonical(cfg)and canonical(c.read(plan['inputs_path']))==canonical(inputs),'Original template/input snapshots differ')
   ports=phase_ports(active);ports['full4'](root,report,'pre');ports['full4'](root,report,'post');identity=report['per_arm'][phase]
   require(canonical(identity['pre_source_binding'])==canonical(identity['post_source_binding'])==canonical(active.fresh.binding()),'Original native current75/all8 source proof differs')
   active.campaign_stats(report,active.stat_snapshot([Path(r['path'])for r in report['pre_full4']['rows']]))
   ports['pages'](report['post_before_pages'],Path(report['post_full4']['rows'][2]['path']));require(report['post_health']['finished_epoch']<=report['post_before_pages']['epoch']<=report['post_full4']['started'],'Original failure-close byte bracket differs')
   existing=ports['existing'];post=existing.health(root,phase+'-post',identity['post_health']);require(post['started_epoch']>=identity['actual_model_terminal_epoch'],'Per-arm posthealth must follow actual native terminal');existing.health(root,report['aggregate_post_health_label'],report['post_health'])
   from native_rms_kernel_journal_v5 import reject_faults
   for label in(phase+'-pre',phase+'-post','post'):
    existing.command_row(root,label+'-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager']);reject_faults((root/(label+'-journal.log')).read_text())
   views,result=ports['phase_binding'](root,phase,m,cfg,inputs,report);require(result['passed']is True and result['requests']==4 and result['finished_epoch']<=identity['actual_model_terminal_epoch'],'Original four normal native results required');require(not (directory/'fidelity-raw.json').exists(),'Recovery requires original failed collector without invented producer proof')
   raw=collect_actual(directory,report['source_plan_sha256']);comparison=active.contract.equivalence(views);pid=active.native_pid(result['terminal'],c.read(directory/'command.json'),False);require(pid==1 and raw['engine_pid']==pid,'Original native/capture PID ownership differs')
   return {'schema':1,'passed':True,'scope':'Recovered current59 mode0 capture component ONLY','original_producer_report_sha256':digest,'original_producer_passed':False,'original_producer_errors':[ERROR],'producer_status_rewritten':False,'original_V3_public_gate_passed':False,'source_generation':59,'materialized_mode':0,'current59_SDK':identity['post_source_binding'],'reader_source_plan_sha256':source,'original_source_plan_sha256':report['source_plan_sha256'],'current_mode0_targets':raw,'current59_four_fresh_comparison':comparison,'original_normal_native_rc':0,'original_posthealth_full4_validated':True,'new_publisher_hash_scan_performed':False,'new_model_or_GPU_execution':False,'full_model_math_qualified':False,'API_qualified':False,'speed_qualified':False}
 finally:require(source_binding()==source and c.sha(root/'report.json')==digest and original.tree(root)==roster,'Original failed producer/source/artifact tree changed during recovery')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external recovery output required');c.write(a.output,finalized_binding(a.root))
