"""Independent actual56 best-HC Direct source difference, lifecycle and histories."""
import argparse,types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import run_source56_best_HC_PLEdirect_v1 as actor
import source61_ple_identity_ports_v1 as identity
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from validate_source56_hc_prefill_tile4_natural_v1 import observations
from source56_best_HC_PLEdirect_contract_v1 import baseline61_binding,HC
c=actor.c;require=actor.require;ROOT=actor.ROOT;HERE=actor.HERE
PLAN=HERE/'source56-best-HC-PLEdirect-reader-source-plan-v1.json'
def source_binding():
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Clean ABBA reader source changed '+name)
 return c.sha(PLAN)

@contextmanager
def configured(report,recipe):
 parent=actor.parent;old_ns,old_phases=parent.NS,parent.PHASES
 try:
  identity.configure('qualifying');parent.NS=actor.NS;parent.PHASES=actor.PHASES;parent.configure(recipe['natural_settings']);parent.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=parent.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);rt=recipe['tool_runtime'];ns['set_runtime'](rt['intake_path'],rt['exact_intake_sha256']);ns['_PORTS']=parent.base.boundary_ports(ns)
  yield SimpleNamespace(**{**vars(actor),**ns})
 finally:parent.NS=old_ns;parent.PHASES=old_phases

def finalized_binding(root):
 root=Path(root).resolve();reader=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');require(not report.get('qualification_pending_CPU_identity',False),'Pending GPU child cannot qualify');roster={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'};require(roster==report['artifacts'],'Actual ABBA byte roster differs')
 require(set(report['results'])==set(actor.PHASES) and len(report['results'])==1 and report['identity_scope']=='whole_matched_arms','One actual56 qualifying best-HC direct owner required')
 recipe=c.read(root/actor.PHASES[0]/'recipe.json')
 try:
  with configured(report,recipe) as active:
   proof=boundary_reader_ports(active)['finalized_binding'](root);arms=[];views=[];recipes=[]
   for phase in actor.PHASES:
    directory=root/phase;r=c.read(directory/'recipe.json');recipes.append(r);rows=c.read(directory/'requests.json');views.append(actor.requests(rows,r['requests']));env=r['native_env'];argv=r['native_argv'];combined=(directory/'engine.combined.log').read_text(encoding='ascii').splitlines()
    require(not any(x.startswith(('PFH60 ','BCOOP ','SBF ','FC38 ','SFD ','PCL ')) for x in combined),'Clean native emitted disabled diagnostic records')
    require(not (directory/'ARM').exists() and not (directory/'NO_CAPTURE_ARM').exists() and not list((directory/'captures').iterdir()) and not list((directory/'host-traces').glob('*')) and not list((directory/'unitrace').glob('*')),'Clean native must have no armed or heavy capture')
    arms.append(dict(phase=phase,PLE_mode=r['PLE_mode'],natural_observations=observations(rows,r),startup_memory_admission=c.read(directory/'memory-admission.json'),expert_profile=r['expert_profile_binding'],cold_cache_qualified=False,physical_IO_bytes_qualified=False))
   require(len(recipes)==1 and recipes[0]['actual_compiled_HC_flags']==HC,'One actual compiled HC2/1/1 owner required');baseline61_binding(recipes[0]);comparison=actor.equivalence(*views);require(comparison==report['H36_OFF_ON_comparison'],'Saved ABBA history comparison differs')
   proof.update(reader_source_plan_sha256=reader,producer_report_sha256=report_sha,source_generation=56,actual56_SDK=active.fresh.binding(),source56_vs61_difference_explicit=True,same_source_three_flag_AB_qualified=False,arms=arms,processes=1,requests_per_process=6,matched_outputs_equal=True,current61_B1_B2_full_histories_equal=True,native_instrumentation_OFF=True,natural_EOS_honored=True,task_completion_qualified=False,cold_cache_qualified=False,physical_IO_bytes_qualified=False,original_fullmath_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False,clean_environment_observed_qualified=False,clean_environment_limitation='Root must separately bind absence of overlapping CPU build/hash/GPU workloads during actual model intervals; this reader validates native flags and boundaries, not all host activity',bounded_actual56_best_HC_PLEdirect_capture_passed=True)
   return proof
 finally:require(source_binding()==reader and c.sha(root/'report.json')==report_sha and actor.closure()==report['source_plan_sha256'] and roster=={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'},'Closing ABBA source/raw association changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public result required');c.write(a.output,finalized_binding(a.root))
