"""Current combined62 two-arm owned/raw reader, independent descriptive math."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import run_native62_hc_rms_ON_v2 as actor
import native62_hc_rms_raw_v2 as raw_reader
from native62_hc_rms_comparison_v2 import independent_metrics
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from validate_source59_mode0_fidelity_v3 import tree
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;PLAN=actor.PLAN

def source_binding():
 before=c.sha(PLAN)
 for n,h in c.read(PLAN)['files'].items():require(c.sha(actor.ROOT/n)==h,'Current combined62 public source changed '+n)
 require(c.sha(PLAN)==before,'Current combined62 public plan changed');return before
@contextmanager
def configured(report,recipe):
 b=actor.c.read(Path(report['_OFF_binding_path']));actor._OFF=(Path(b['path']),b['sha256']);p=actor.parent;p.configure(recipe['natural_settings']);p.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=p.fresh;ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);runtime=recipe['tool_runtime'];ns['set_runtime'](runtime['intake_path'],runtime['exact_intake_sha256']);ns['_PORTS']=p.base.boundary_ports(ns);active=SimpleNamespace(**{**vars(actor),**ns});require(active.PHASES==actor.PHASES and active.Protocol is actor.Protocol and active.contract.requests is actor.requests and active.life.contract.native_command is actor.native_command,'Exact current private namespace/callback required');yield active

def finalized_binding(root,expected_report_sha256):
 root=Path(root).resolve();source=source_binding();digest=c.sha(root/'report.json');require(digest==expected_report_sha256,'Exact original current62 producer report required');report=c.read(root/'report.json');roster=tree(root);require(roster==report['artifacts']and report['passed']is True and report['errors']==[],'Authentic successful two-arm source/currentbyte campaign required');phase0=actor.PHASES[0];r0=c.read(root/phase0/'recipe.json');report['_OFF_binding_path']=str(root/phase0/'OFF-component-binding.json');proof=None
 try:
  with configured(report,r0)as active:
   require(active.fresh.binding()['source_generation']==62,'Genuine combined62 SDK required');proof=boundary_reader_ports(active)['finalized_binding'](root);require(set(report['results'])==set(actor.PHASES),'Exact both phases required');recipes=[];targets={};views=[]
   for index,phase in enumerate(actor.PHASES):
    d=root/phase;recipe=c.read(d/'recipe.json');recipes.append(recipe);require(recipe['engine_generation']==62 and recipe['materialized_mode']==0 and recipe['NUM33_enabled']is True and recipe['PTI_enabled']is False and recipe['cards']==[0,1]and recipe['stages']==[[0,0,24],[1,24,48]]and recipe['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']=='1','Exact phase/source/topology/math flag required');result=c.read(d/'result.json');pid=active.native_pid(result['terminal'],c.read(d/'command.json'),False);require(type(pid)is int and pid==1,'Actual directexec PID1 required')
    require(all(recipe['native_env'].get(k)=='0'for k in('STRATA_QSA_INDEXER_MATERIALIZED_HALF','STRATA_CRITICAL_PATH_TRACE','STRATA_BATCH_FIDELITY_DIAG','STRATA_BATCH_PREFILL_LAYER_COOPERATE','STRATA_ORIGINAL_MTP53'))and recipe['native_env']['SYCL_UR_TRACE']=='2','Exact diagnostic flags required');require(not list((d/'host-traces').glob('*'))and not list((d/'unitrace').glob('*'))and not list((d/'qsa').glob('*')),'Disabled observers emitted raw artifacts');raw=raw_reader.collect(d,report['source_plan_sha256'],actor);require(raw['source_generation']==62 and raw['native_numerical_observer_field_count']==33 and raw['extra44_core_RMS_fields_observed']is False and canonical(raw)==canonical(c.read(d/'fidelity-raw.json')),'Fresh recollected original33/full49/P30 targets differ');comparison=independent_metrics(raw);require(canonical(comparison)==canonical(c.read(d/'numerical-comparison.json')),'Recomputed baseline/independent metrics differ');targets[phase]=raw;views.append(actor.requests(c.read(d/'requests.json'),recipe['requests']))
   require(len(recipes)==1 and actor._OFF is not None,'Exactly newON with authenticated savedOFF required');off=c.read(actor._OFF[0]);require(c.sha(actor._OFF[0])==actor._OFF[1]and canonical(off)==canonical(c.read(root/phase0/'OFF-component.snapshot.json')),'Original OFFcomponent snapshot/bytes changed');left=off['actual_phase_recipe'];right=recipes[0];actor.matched_recipe_gate(left,right)
   from recover_native62_hc_rms_OFF_v1 import finalized_binding as recover
   revalidated=recover(off['source_root']);require(canonical(revalidated)==canonical(off),'Original completedOFF exactsource/raw/lifecycle proof changed')
   require(canonical(actor.equivalence(*views))==canonical(report['H36_OFF_ON_comparison']),'Original repeat/coherence result differs');proof.update(source_generation=62,materialized_mode=0,current_SDK=active.fresh.binding(),producer_report_sha256=digest,reader_source_plan_sha256=source,current_mode0_targets=targets,new_native_processes=1,new_requests=4,OFF_actualcomponent_reused=True,failed_OFF_parent_preserved=True,OFF_component_sha256=actor._OFF[1],baseline_original59_tensor_bitwise=True,ON_independent_full48_comparison=c.read(root/actor.PHASES[0]/'numerical-comparison.json'),native_RMS3_actual_MAIN_observed=False,numeric_pass_assigned=False,full_model_math_qualified=False,API_qualified=False,speed_qualified=False)
   return proof
 finally:require(source_binding()==source and c.sha(root/'report.json')==digest and tree(root)==roster,'Closing actual source/report/full artifact tree changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external admission output required');c.write(a.output,finalized_binding(a.root,a.report_sha256))
