"""Independent current59 native ownership/raw reader; no math-reference promotion."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import run_source59_mode0_fidelity_v1 as actor
import source59_mode0_fidelity_raw_v1 as raw_reader
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;ROOT=actor.ROOT;HERE=actor.HERE
PLAN=HERE/'source59-mode0-fidelity-capture-source-plan-v1.json'
def source_binding():
 source=c.sha(PLAN)
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Current59 fidelity public source changed '+name)
 require(c.sha(PLAN)==source,'Current59 fidelity plan changed');return source

def tree(root):
 value={}
 for p in Path(root).rglob('*'):
  require(not p.is_symlink(),'Symlink in original current59 artifact tree')
  if p.is_file()and p.name!='report.json':value[str(p.relative_to(root))]=c.sha(p)
 return value

@contextmanager
def configured(report,recipe):
 p=actor.parent;old_ns,old_phases=p.NS,p.PHASES
 try:
  p.NS=actor.NS;p.PHASES=actor.PHASES;p.configure(recipe['natural_settings']);p.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=p.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);runtime=recipe['tool_runtime'];ns['set_runtime'](runtime['intake_path'],runtime['exact_intake_sha256']);ns['_PORTS']=p.base.boundary_ports(ns)
  active=SimpleNamespace(**{**vars(actor),**ns});require(active.PHASES==actor.PHASES and active.Protocol is actor.Protocol and active.contract.requests is actor.requests and active.life.contract.native_command is actor.native_command,'Exact active current59 namespace/callback required')
  yield active
 finally:p.NS=old_ns;p.PHASES=old_phases

def finalized_binding(root,expected_report_sha256):
 root=Path(root).resolve();require(root.is_dir()and not root.is_symlink(),'Canonical original current59 run root required');source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');require(type(expected_report_sha256)is str and report_sha==expected_report_sha256,'Exact authenticated original current59 producer report SHA required');roster=tree(root)
 require(roster==report['artifacts'],'Original current59 full artifact tree differs');require(report['passed']is True and report['errors']==[]and report['SDK_root']==str(actor.SDK)and report['SDK_receipt_sha256']==actor.SDK_SHA,'Authentic successful current59 report required')
 phase=actor.PHASES[0];recipe=c.read(root/phase/'recipe.json')
 try:
  with configured(report,recipe)as active:
   proof=boundary_reader_ports(active)['finalized_binding'](root)
   require(set(report['results'])=={phase}and len(recipe['requests'])==4 and recipe['engine_generation']==59 and recipe['materialized_mode']==0 and recipe['NUM33_enabled']is True and recipe['PTI_enabled']is False and recipe['cards']==[0,1]and recipe['stages']==[[0,0,24],[1,24,48]],'Exact single current59 paired four-row descriptor required')
   directory=root/phase;result=c.read(directory/'result.json');cmd=c.read(directory/'command.json');pid=active.native_pid(result['terminal'],cmd,False);require(type(pid)is int and pid==1,'Actual directexec isolated native namespace PID1 required')
   require(all(recipe['native_env'].get(k)=='0'for k in ('STRATA_QSA_INDEXER_MATERIALIZED_HALF','STRATA_CRITICAL_PATH_TRACE','STRATA_BATCH_FIDELITY_DIAG','STRATA_BATCH_PREFILL_LAYER_COOPERATE','STRATA_ORIGINAL_MTP53')) and recipe['native_env']['SYCL_UR_TRACE']=='2','Actual current0 diagnostic ENV differs')
   require(not list((directory/'host-traces').glob('*'))and not list((directory/'unitrace').glob('*'))and not list((directory/'qsa').glob('*')),'Disabled H36/PTI/QSA3 emitted artifacts')
   raw=raw_reader.collect(directory,report['source_plan_sha256']);require(canonical(raw)==canonical(c.read(directory/'fidelity-raw.json')),'Actual current59 saved raw collection differs')
   views=actor.requests(c.read(directory/'requests.json'),recipe['requests']);require(canonical(actor.equivalence(views))==canonical(report['H36_OFF_ON_comparison']),'Original four-repeat coherence result differs')
   proof.update(source_generation=59,materialized_mode=0,current59_SDK=active.fresh.binding(),producer_report_sha256=report_sha,reader_source_plan_sha256=source,engine_pid=pid,current_mode0_targets=raw,native_processes=1,requests=4,full49_P30_NUM33_owning_UR_targets_recollected=True,full_model_math_qualified=False,API_qualified=False,concurrency_qualified=False,speed_qualified=False,H36_OFF_ON_equivalence_passed=False,observer_OFF_ON_comparison_performed=False)
   return proof
 finally:require(source_binding()==source and c.sha(root/'report.json')==report_sha and tree(root)==roster,'Closing current59 public source/report/raw tree changed')

def captured_binding(root,expected_report_sha256):
 root=Path(root).resolve();proof=finalized_binding(root,expected_report_sha256);report=c.read(root/'report.json');directory=root/actor.PHASES[0]
 result={'capture_root':str(directory),'source_generation':59,'materialized_mode':0,'observer_enabled':True,'NUM33_enabled':True,'QSA3_enabled':False,'engine_pid':1,'native_observer_field_count':33,'extra44_core_RMS_fields_observed':False,'plan_sha256':report['plan_sha256'],'binding_sha256':report['source_plan_sha256'],'requests':proof.pop('current_mode0_targets')['requests'],'current59_SDK':proof['current59_SDK'],'public_actor_admission':proof,'full_model_math_qualified':False}
 for role in('command','result','phase_receipt'):
  p=directory/(role+'.json');result[role+'_path']=str(p);result[role+'_sha256']=c.sha(p)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external current59 public output required');c.write(a.output,finalized_binding(a.root,a.report_sha256))
