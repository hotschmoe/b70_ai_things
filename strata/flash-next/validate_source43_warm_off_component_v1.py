#!/usr/bin/env python3
"""Named closed OFF component; failed whole parent is never a successful view."""
import ast,copy
from types import SimpleNamespace
from pathlib import Path
import run_source43_warm_pti_v3 as actor
import validate_source43_bounded_pti_v1 as reader
import validate_source43_warm_pti_v3 as warm
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;PHASE=actor.PHASES[0]

def helpers(historical_actor):
 ns={**vars(reader),'actor':historical_actor};node=copy.deepcopy(next(n for n in ast.parse(Path(reader.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='ports'));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[original-OFF-phase-helpers]','exec'),ns);return ns['ports']()

def parent_scope(report):
 require(type(report['passed'])is bool and report['passed']is False and type(report['errors'])is list and report['errors'],'Actual failed parent and original errors must remain failed')
 require(PHASE in report['results']and PHASE in report['per_arm'],'Actual completed OFF component required')
 return {'original_parent_passed':False,'original_parent_errors':report['errors'],'whole_parent_qualified':False}

def body(root,expected_report_sha256):
 own_plan=actor.HERE/'source43-warm-pti-source-plan-v4.json';own_source=c.sha(own_plan)
 for name,want in c.read(own_plan)['files'].items():require(c.sha(actor.ROOT/name)==want,'Exact OFF-component reader source changed '+name)
 root=Path(root).resolve();require(c.sha(root/'report.json')==expected_report_sha256,'Exact original failed parent SHA required');report=c.read(root/'report.json');scope=parent_scope(report);source=actor.closure();ns=reader.base.ports();roster=ns['tree'](root)
 require(canonical(roster)==canonical(report['artifacts'])and report['source_plan_sha256']==source==c.sha(root/'source-plan.snapshot.json')and c.sha(root/'producer.snapshot.py')==c.sha(actor.__file__),'Original component source/producer/whole artifact roster differs')
 recipe=c.read(root/PHASE/'recipe.json');plan=c.read(root/'plan.snapshot.json');require(c.sha(root/'plan.snapshot.json')==report['plan_sha256'],'Original corpus plan differs');m=c.read(root/'prepared.snapshot.json');cfg=c.read(root/'config.snapshot.json');inputs=c.read(root/'inputs.json')
 require(c.sha(root/'prepared.snapshot.json')==plan['template_prepared_sha256']and c.sha(root/'config.snapshot.json')==plan['template_config_sha256']and canonical(inputs)==canonical(c.read(plan['inputs_path'])),'Exact original prepared/config/independent input snapshots differ')
 # Recipe reconstruction is software only: explicitly saved runtime/owner inputs.
 # It does not invoke the current SDK, upload, masked or fixture runtime readers.
 runtime=recipe['tool_runtime'];private={**vars(actor),'_RUNTIME':runtime,'_CASE':(report['role_input'],report['producer_pid'],report['producer_start_ticks'])}
 private['tool']=SimpleNamespace(**{**vars(actor.fresh_tool),'LIB_SHA':runtime['injection']['sha256'],'TOOL_SHA':runtime['tool']['sha256']})
 node=copy.deepcopy(next(n for n in ast.parse(Path(actor.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='phase_recipe'));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[saved-OFF-software-recipe-only]','exec'),private)
 historical_actor=SimpleNamespace(**{**vars(actor),'phase_recipe':private['phase_recipe']});ns=helpers(historical_actor)
 current=dict(m);current['source']=str(actor.intake.SDK/'source');tname,tcmd=actor.life.contract.tokenizer_command(root,current,report['producer_pid'],source,actor.life.__file__);require(canonical(tcmd)==canonical(c.read(root/'tokenizer.command.json')),'Original tokenizer command changed');tr=c.read(root/'tokenizer.receipt.json');actor.life.contract.inspection(tr['terminal'],tname,tcmd,current['runtime']['image'],source,True);actor.life.terminal(tr['terminal']['State']);require(tr['error']is None and type(tr['return_code'])is int and tr['return_code']==0 and tr['removed']is True and tr['removal']['recovery_used']is False and c.absent(tname),'Original metadata retirement failed')
 for stream in('stdout','stderr'):ns['existing'].pipe(tr[stream],root/('tokenizer.'+stream+'.log'))
 speed=actor.life.match_speed_inputs(report['speed_input_join']['path'],inputs,m);require(canonical(speed)==canonical(report['speed_input_join']),'Original speed input association changed')
 identity=report['per_arm'][PHASE];ns['full4'](root,identity,'pre',PHASE+'-');ns['full4'](root,identity,'post',PHASE+'-');existing=ns['existing'];pre=existing.health(root,PHASE+'-pre',report[PHASE+'-pre_health']);post=existing.health(root,PHASE+'-post',identity['post_health']);require(identity['pre_after_pages']['epoch']<=pre['started_epoch']and post['started_epoch']>=identity['actual_model_terminal_epoch'],'Original OFF identity/health/terminal order differs')
 from native_rms_kernel_journal_v5 import reject_faults
 for label in(PHASE+'-pre',PHASE+'-post'):
  existing.command_row(root,label+'-journal',['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager']);reject_faults((root/(label+'-journal.log')).read_text())
 views,result=ns['phase_binding'](root,PHASE,m,cfg,inputs,report);require(result['finished_epoch']<=identity['actual_model_terminal_epoch']and canonical(result)==canonical(report['results'][PHASE]),'Original OFF result/terminal join differs');pid=actor.native_pid(result['terminal'],c.read(root/PHASE/'command.json'),False)
 files=sorted((root/PHASE/'host-traces').glob('host-trace36-*.jsonl'));fresh=actor.collector.collect(files,root/PHASE/'engine.combined.log',source,[r['raw']for r in c.read(root/PHASE/'requests.json')],[[0,0,48]],pid);require(canonical(fresh)==canonical(c.read(root/PHASE/'host-coverage.json'))and canonical(warm.graph_binding(root/PHASE))==canonical(c.read(root/PHASE/'warm-graph-proof.json')),'Original OFF H36/graph coverage differs')
 require(c.sha(own_plan)==own_source and actor.closure()==source and ns['tree'](root)==roster and c.sha(root/'report.json')==expected_report_sha256,'Original failed parent source/report/artifacts changed')
 return {'passed':True,'scope':'completed OFF component only; failed whole parent preserved','root':str(root),'report_sha256':expected_report_sha256,'source_plan_sha256':source,'component_reader_source_plan_sha256':own_source,'phase':PHASE,'request_views':views,'config_sha256':c.sha(root/'config.snapshot.json'),'source_generation':43,'materialized_mode':0,'component_original_recorded_phase_gates_passed':True,'full_model_math_qualified':False,'API_qualified':False,'historical_metadata_only':True,'current_SDK_admission_reexecuted':False,'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'fresh_public_full4_rehashed':False,'new_arm_independent_current_admission_and_fresh_full4_health_required':True,**scope}

def finalized_binding(root,expected_report_sha256):
 return body(root,expected_report_sha256)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external OFF component output required');c.write(a.output,finalized_binding(a.root,a.report_sha256))
