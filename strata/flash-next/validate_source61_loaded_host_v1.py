"""Public native2 cache reader; original model/lifecycle gates and exact batch partition."""
import argparse, ast, copy, json, re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import run_source61_loaded_host_v1 as actor
import validate_source47_grouped_f16_model_v2 as boundary
from serial37_canonical_json_v3 import canonical
c=actor.c; require=actor.require; ROOT=actor.ROOT; HERE=actor.HERE
PLAN=HERE/'source61-loaded-host-reader-source-plan-v1.json'

def source_binding():
 for name,want in c.read(PLAN)['files'].items(): require(c.sha(ROOT/name)==want,'New native2 reader source changed '+name)
 return c.sha(PLAN)

def batch_partition(directory, rows, recipe, lines, combined):
 require(lines==[line for line in combined.decode('ascii').splitlines() if actor.Protocol.is_protocol(line)],'Exact observed batch combined/protocol partition differs')
 require(sum(line.startswith('READY') for line in lines)==1,'Unique actual native READY required')
 stdin=[json.loads(line) for line in (directory/'native.stdin.jsonl').read_text(encoding='ascii').splitlines()]
 expected=[actor.command_for(row)+'\n' for row in recipe['requests']]+['QUIT\n']
 require([row['packet'] for row in stdin]==expected and [row['seq'] for row in stdin]==list(range(len(recipe['requests'])+1)) and all(type(row['monotonic_ns']) is int for row in stdin) and [row['monotonic_ns'] for row in stdin]==sorted(row['monotonic_ns'] for row in stdin),'Exact original complete native send sequence differs')
 joined=[]
 for cohort in sorted({row['cohort'] for row in recipe['requests']}):
  receipt=c.read(directory/('cohort-%d.json'%cohort)); events=receipt['events']; joined.extend(row['line'] for row in events)
  scheduled=[row for row in recipe['requests'] if row['cohort']==cohort]
  require(receipt['cohort']==cohort and receipt['commands']==[actor.command_for(row) for row in scheduled],'Original cohort command identity differs')
  for want in scheduled:
   raw=rows[want['ordinal']-1]['raw']; own=[(x['line'],x['host_monotonic_seconds']) for x in events if actor.fields(x['line']).get('rid')==str(want['rid']) and x['line'].startswith(('T ','BT ','DONE ','BADM ','BDONE '))]
   require(raw['stdout']==[line for line,_ in own] and raw['stdout_monotonic_seconds']==[stamp for _,stamp in own],'Original current owner partition/timing differs')
 # Source observers can emit final release lines after the last BDONE; startup INFO/SBF
 # likewise precede READY. Cohort intervals must be one contiguous raw protocol span.
 first=lines.index(joined[0]);require(lines[first:first+len(joined)]==joined,'Original bounded cohort events omitted/reordered/duplicated')
 require(not any(line.startswith(('T ','BT ','DONE ','BADM ','BDONE ','ERR','FATAL')) for line in lines[:first]+lines[first+len(joined):]),'Unassociated native token/request/terminal/error outside current cohorts')
 return True

# Preserve original boundary reader predicates; replace only its serial partition
# assumption with the exact current co-admission partition above.
def reader_ports(active):
 result=boundary.boundary_reader_ports(active)
 original=boundary.original.base.original
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='phase_binding'))
 changes=0
 for item in ast.walk(node):
  if isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='require' and len(item.args)==2 and isinstance(item.args[1],ast.Constant) and item.args[1].value=='Actual exact protocol/raw request partition differs':
   item.args[0]=ast.parse('batch_partition(directory,rows,recipe,lines,combined)',mode='eval').body;changes+=1
 require(changes==1,'Exactly one original serial partition predicate required')
 result['batch_partition']=batch_partition
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[explicit-native2-partition-port]','exec'),result)
 return result

@contextmanager
def configured(report,recipe):
 parent=actor.parent; old_ns=parent.NS; old_phases=parent.PHASES
 try:
  parent.NS=actor.NS;parent.PHASES=actor.PHASES;parent.configure(recipe['natural_settings']);parent.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=parent.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);runtime=recipe['tool_runtime'];ns['set_runtime'](runtime['intake_path'],runtime['exact_intake_sha256']);ns['_PORTS']=parent.base.boundary_ports(ns)
  active=SimpleNamespace(**{**vars(actor),**ns});require(active.PHASES==actor.PHASES and active.contract.requests is actor.requests and active.Protocol is actor.Protocol,'Current native2 namespace/contract/protocol differs')
  yield active
  require(actor.closure()==report['source_plan_sha256'],'Closing current native2 actor changed')
 finally:parent.NS=old_ns;parent.PHASES=old_phases

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');artifacts={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'}
 require(artifacts==report['artifacts'],'Original default-scheduler artifacts differ');recipe=c.read(root/actor.PHASES[0]/'recipe.json')
 try:
  with configured(report,recipe) as active:
   proof=reader_ports(active)['finalized_binding'](root);require(set(report['results'])==set(actor.PHASES) and len(report['results'])==1,'Two exact serial/overlap owned phases required');observations=[];recipes=[]
   for phase in actor.PHASES:
    directory=root/phase;current=c.read(directory/'recipe.json');recipes.append(current);observations.append(actor.requests(c.read(directory/'requests.json'),current['requests']))
    require(current['native_env']['STRATA_TRACE']=='1' and current['cooperative_diagnostic_trace'] is True and current['budget_runtime_ledger_available'] is False and 'STRATA_BATCH_DECODE_SHARE' not in current['native_env'] and current['native_env']['STRATA_BATCH_FIDELITY_ARM']=='/results/NO_CAPTURE_ARM' and current['native_env']['STRATA_FULL_CACHE_OBSERVER38']=='0' and current['engine_generation']==61 and current['native_env']['STRATA_VERIFY_MIRROR_STAGE_T1']=='0' and current['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and current['PTI_enabled'] is False and current['observer_enabled'] is False and current['raw_observer_armed'] is False and current['cards']==[0,1] and current['stages']==[[0,0,24],[1,24,48]],'Exact current52 default scheduling/no raw capture/pair profile required')
    require(current['native_argv'][current['native_argv'].index('--batch')+1]=='2' and not list((directory/'captures').iterdir()) and not (directory/'NO_CAPTURE_ARM').exists() and not (directory/'ARM').exists() and not list((directory/'host-traces').glob('*')) and not list((directory/'unitrace').glob('*')),'Exact native2 and absent heavy observer required')
   require(len(recipes)==1 and recipes[0]['native_env']['STRATA_PREFILL_HOST60']=='1' and recipes[0]['native_env']['STRATA_BATCH_PREFILL_LAYER_COOPERATE']==recipes[0]['native_env']['STRATA_BATCH_PREFILL_SINGLE_WINDOW']=='1' and len(recipes[0]['requests'])==4,'One fixed diagnostic1/cap1/cooperation1 loaded owner required')
   comparison=actor.equivalence(*observations);require(canonical(comparison)==canonical(report['H36_OFF_ON_comparison']),'Original bounded scheduler observation differs')
   proof.update(normal_scheduler_observations=observations,loaded_host_comparison=comparison,source_generation=61,actual61_SDK=active.fresh.binding(),reader_source_plan_sha256=source,producer_report_sha256=report_sha,native_processes=1,native_scheduled_streams=4,native2_simultaneous_coherence_qualified=False,coadmission_qualification_open=True,budget_runtime_accounting_qualified=False,cancellation_usability_qualified=False,clean_performance_qualified=False,bounded61_loaded_host_localization_passed=True,raw_capture_armed=False,full_cache_qualified=False,original_fullmath_qualified=False,API_qualified=False,shelf_qualified=False,HTTP_latency_qualified=False,fairness_stability_qualified=False,speed_promotion_qualified=False)
   return proof
 finally:require(source_binding()==source and c.sha(root/'report.json')==report_sha and artifacts=={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'},'Closing current source/artifact binding differs')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();require(not args.output.exists(),'New external public result required');c.write(args.output,finalized_binding(args.root))
