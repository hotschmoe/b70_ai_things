"""Authenticate saved completedOFF of failed62 producer; no native rerun/status rewrite."""
import argparse,ast,copy
from pathlib import Path
import validate_native62_hc_rms_fidelity_v1 as original
import native62_hc_rms_raw_v1 as raw_reader
from native62_hc_rms_comparison_v2 import baseline
from serial37_canonical_json_v3 import canonical
c=original.c;require=original.require;actor=original.actor
PLAN=actor.HERE/'native62-hc-rms-OFF-recovery-source-plan-v1.json'
REPORT_SHA='72e1867f56f2748483f1529cb33ff00c9075bf7069c6cfa0c50cda796bc201d5'
PHASE='native62-hc-rms-OFF'
def source_binding():
 before=c.sha(PLAN)
 for n,h in c.read(PLAN)['files'].items():require(c.sha(actor.ROOT/n)==h,'Recovery source changed '+n)
 require(c.sha(PLAN)==before,'Recovery source plan changed');return before

def phase_ports(active):
 ports=original.boundary_reader_ports(active)
 import validate_source41_model_cli_v5 as grammar
 node=copy.deepcopy(next(n for n in ast.parse(Path(grammar.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='phase_binding'));count=0
 for n in ast.walk(node):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='canonical'and n.args and ast.unparse(n.args[0])=="report['results'][phase]":n.args[0]=ast.Name(id='result',ctx=ast.Load());count+=1
 require(count==1,'Exactly failedcollector missing report-result row join required');ports['original_result_receipt_join']=True;exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[actualmissing-parentrow-only]','exec'),ports);return ports

def recovery_gate(report,digest,roster):
 require(digest==REPORT_SHA and report['passed']is False and report['errors']==['ValueError: Actual combinedSDK must be configured']and report['results']=={}and set(report['per_arm'])=={PHASE}and report['source_plan_sha256']==c.sha(actor.PLAN)and roster==report['artifacts'],'Exact preservedfailed post-native namespace attempt required')

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();digest=c.sha(root/'report.json');report=c.read(root/'report.json');roster=original.tree(root);recovery_gate(report,digest,roster);directory=root/PHASE;recipe=c.read(directory/'recipe.json')
 try:
  # Configure the actually imported canonical actor explicitly from authentic
  # saved producer metadata before any field_contract/raw-consumer call.
  with original.configured(report,recipe)as active:
   require(actor.parent.fresh.SDK==Path(report['SDK_root'])and actor.parent.fresh.RECEIPT_SHA==report['SDK_receipt_sha256'],'Actual imported currentSDK namespace not configured');require(actor.closure()==report['source_plan_sha256']and c.sha(root/'source-plan.snapshot.json')==report['source_plan_sha256']and c.sha(root/'producer.snapshot.py')==c.sha(actor.__file__),'Original actualproducer/source snapshots differ');require(recipe['engine_generation']==62 and recipe['materialized_mode']==0 and recipe['NUM33_enabled']is True and recipe['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']=='0','Exact completedOFF current62 math descriptor required');ports=phase_ports(active);ports['full4'](root,report,'pre');ports['full4'](root,report,'post');arm=report['per_arm'][PHASE]
   require(canonical(arm['pre_source_binding'])==canonical(arm['post_source_binding'])==canonical(active.fresh.binding()),'Original actual62 current source/all8 beforeafter differ');active.campaign_stats(report,active.stat_snapshot([Path(r['path'])for r in report['pre_full4']['rows']]));ports['pages'](arm['pre_pages'],Path(report['pre_full4']['rows'][2]['path']));ports['pages'](arm['post_pages'],Path(report['post_full4']['rows'][2]['path']));ports['pages'](report['post_before_pages'],Path(report['post_full4']['rows'][2]['path']));require(report['post_health']['finished_epoch']<=report['post_before_pages']['epoch']<=report['post_full4']['started'],'Original posthealth/page/fourhash chronology differs');existing=ports['existing'];health=existing.health(root,PHASE+'-post',arm['post_health']);require(health['started_epoch']>=arm['actual_model_terminal_epoch'],'Actual perarm posthealth before native terminal');existing.health(root,report['aggregate_post_health_label'],report['post_health'])
   from native_rms_kernel_journal_v5 import reject_faults
   for label in(PHASE+'-pre',PHASE+'-post','post'):
    existing.command_row(root,label+'-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager']);reject_faults((root/(label+'-journal.log')).read_text())
   plan=c.read(root/'plan.snapshot.json');require(c.sha(root/'plan.snapshot.json')==report['plan_sha256'],'Original inputplan changed');m=c.read(root/'prepared.snapshot.json');cfg=c.read(root/'config.snapshot.json');inputs=c.read(root/'inputs.json');require(c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256']and canonical(c.read(Path(plan['template_path'])/'prepared.json'))==canonical(m)and canonical(c.read(Path(plan['template_path'])/'server-config.json'))==canonical(cfg)and canonical(c.read(plan['inputs_path']))==canonical(inputs),'Original actualtemplate/config/input snapshots changed')
   views,result=ports['phase_binding'](root,PHASE,m,cfg,inputs,report);require(result['passed']is True and result['requests']==4 and result['finished_epoch']<=arm['actual_model_terminal_epoch'],'Original four requests/normalterminal required');pid=active.native_pid(result['terminal'],c.read(directory/'command.json'),False);require(type(pid)is int and pid==1,'Actual capture/native namespace owner differs');require(not(directory/'fidelity-raw.json').exists()and not(directory/'numerical-comparison.json').exists(),'Original failedcollector proof must remain absent');raw=raw_reader.collect(directory,report['source_plan_sha256']);comparison=baseline(raw);return {'schema':1,'passed':True,'scope':'Authenticated saved actual62 completedOFF component; failedaggregate preserved','original_producer_report_sha256':digest,'original_producer_passed':False,'original_producer_errors':report['errors'],'producer_status_rewritten':False,'original_aggregate_qualification_passed':False,'source_generation':62,'materialized_mode':0,'explicit_sqrt_divide':0,'source_plan_sha256':source,'original_actor_source_plan_sha256':report['source_plan_sha256'],'SDK_root':report['SDK_root'],'SDK_receipt_sha256':report['SDK_receipt_sha256'],'currentSDK':active.fresh.binding(),'actual_normal_native_rc':0,'saved_posthealth_full4_validated':True,'new_full_model_hash_scan_performed':False,'GPU_or_model_rerun':False,'baseline_original59':comparison,'actual_OFF_targets':raw,'actual_phase_recipe':recipe,'source_root':str(root),'numeric_pass_assigned':False,'full_model_math_qualified':False,'API_qualified':False,'speed_qualified':False}
 finally:require(source_binding()==source and c.sha(root/'report.json')==digest and original.tree(root)==roster,'Closing original failed source/report/artifact tree changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external OFFrecovery output required');c.write(a.output,finalized_binding(a.root))
