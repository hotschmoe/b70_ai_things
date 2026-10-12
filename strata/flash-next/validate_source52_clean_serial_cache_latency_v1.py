"""Independent actual52 serial cache reader; preserved original model boundary."""
import argparse,copy,math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_source52_clean_serial_cache_latency_v1 as actor
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;HERE=actor.HERE;ROOT=actor.ROOT;PLAN=HERE/'source52-clean-serial-cache-reader-source-plan-v1.json'

def source_binding():
 value=c.read(PLAN)
 for name,want in value['files'].items():require(c.sha(ROOT/name)==want,'Serial cache public reader source changed '+name)
 return c.sha(PLAN)

@contextmanager
def configured_actor(report,recipe):
 # Replay the producer's explicit private namespace setup; no __getattr__ assumption,
 # no old actor code/global callback is replaced. All changes are process-local intake.
 p=actor.parent;old_ns=p.NS;old_fixture=(actor.FIXTURE,actor.FIXTURE_SHA)
 try:
  p.NS=actor.NS;actor.FIXTURE=Path(recipe['cache_fixture_path']).resolve();actor.FIXTURE_SHA=recipe['cache_fixture_sha256'];p.configure(recipe['natural_settings']);p.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=p.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=recipe['tool_runtime'];ns['set_runtime'](saved['intake_path'],saved['exact_intake_sha256']);ns['_PORTS']=p.base.boundary_ports(ns)
  active=SimpleNamespace(**{**vars(actor),**ns});require(active.__file__==actor.__file__ and active.contract.requests is actor.requests and active.contract.equivalence is actor.equivalence,'Actual serial actor private namespace/contract differs');require(active.fresh is p.fresh and active.fresh.SDK is not None,'Actual source52 configured before reader')
  actor.fixture()
  try:yield active
  finally:require(actor.closure()==report['source_plan_sha256'],'Closing original actor source/fixture changed')
 finally:p.NS=old_ns;actor.FIXTURE,actor.FIXTURE_SHA=old_fixture

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Original actual serial cache roster differs');recipe=c.read(root/actor.parent.PHASES[0]/'recipe.json')
 try:
  with configured_actor(report,recipe)as active:
   result=boundary_reader_ports(active)['finalized_binding'](root);arms=[];views=[];recipes=[]
   for index,phase in enumerate(active.PHASES):
    directory=root/phase;r=c.read(directory/'recipe.json');recipes.append(r);require(type(r['engine_generation'])is int and r['engine_generation']==52 and r['native_env']['STRATA_COPY_ENGINE']=='0'and r['native_env']['STRATA_PF_GROUPED_F16']=='0'and r['native_env']['STRATA_PF_ESIMD_F16']=='0'and r['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and r['PTI_enabled']is False and all(r['native_env'].get(k)=='0'for k in('STRATA_FIDELITY_DIAG','STRATA_FIDELITY_DIAG_ACTIVATIONS','STRATA_PREFIX_DIAG','STRATA_PREFIX_LIFECYCLE_DIAG'))and r['observer_enabled']is False and r['cards']==[0,1]and r['stages']==[[0,0,24],[1,24,48]],'Actual bounded source52 pair/plain compute route required');require(r['native_argv'][r['native_argv'].index('--prompt-cache')+1]=='3'and r['native_argv'][r['native_argv'].index('--conversation-cache-mib')+1]=='0','Exact basic prompt-only cache profile required');rows=c.read(directory/'requests.json');v=actor.requests(rows,r['requests']);views.append(v);observations=v;require([o['actual_reused']for o in observations]==([0]*6 if index==0 else[0,1536,1536,1536,1536,1536]),'Actual complete cold/hit reuse pattern differs');
    for indexes in((0,1,4),(2,3,5)):
     first=observations[indexes[0]]
     for i in indexes[1:]:require(observations[i]['output_ids']==first['output_ids']and observations[i]['finish']==first['finish'],'Within-arm exact role repeats differ')
    require(not list((directory/'captures').glob('*'))and not (directory/'ARM').exists()and not list((directory/'host-traces').glob('*'))and not list((directory/'unitrace').glob('*')),'Disabled heavy observer emitted files');asset=r['expert_profile_binding'];require(actor.parent.profile_binding(asset['path'],asset['sha256'])==asset,'Current source52 expert profile changed');arms.append({'phase':phase,'requests':observations})
   require(recipes[0]['native_argv']==recipes[1]['native_argv']and canonical(recipes[0]['native_env'])==canonical(recipes[1]['native_env']),'Only request fresh state may differ between arms');comparison=actor.equivalence(*views);require(canonical(comparison)==canonical(report['H36_OFF_ON_comparison']),'Original serial cache comparison result differs');proof=active.fresh.binding();result.update(arms=arms,actual52_SDK=proof,source_generation=52,reader_source_plan_sha256=source,producer_report_sha256=report_sha,requests_per_arm=6,native_processes=2,instrumented_cache_diagnostic=False,clean_latency_qualified=False,native_capture_free_latency_observed=True,clean_HTTP_latency_qualified=False,H36_OFF_ON_equivalence_passed=False,first_heads_observed=False,stage_body_coverage_observed=False,cold_hit_output_IDs_equal=True,bounded_basic_cache_passed=True,full_cache_qualified=False,native_concurrency_qualified=False,original_fullmath_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False);return result
 finally:require(source_binding()==source and c.sha(root/'report.json')==report_sha and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing actual reader/source/raw roster differs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
