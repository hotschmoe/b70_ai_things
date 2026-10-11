"""Native47 OFF/ON model comparison; no original full-math authority."""
import ast,copy
from pathlib import Path
import run_source47_grouped_f16_model_v1 as actor
import validate_source43_bounded_pti_v1 as original
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

def finalized_binding(root):
 root=Path(root);report=c.read(root/'report.json');actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=c.read(root/actor.PHASES[0]/'recipe.json')['tool_runtime'];actor.set_runtime(saved['intake_path'],saved['exact_intake_sha256'])
 ns={**vars(original),'actor':actor};node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='ports'));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[original-protected-native-model-public]','exec'),ns);result=ns['ports']()['finalized_binding'](root)
 arms=[];observations=[]
 for index,phase in enumerate(actor.PHASES):
  directory=root/phase;recipe=c.read(directory/'recipe.json');require(recipe['engine_generation']==47 and recipe['materialized_mode']==0 and recipe['native_env']['STRATA_PF_GROUPED_F16']==str(index)and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['native_argv'][recipe['native_argv'].index('--prefill')+1]=='64','Exact native47 grouped OFF/ON and traceOFF required');rows=c.read(directory/'requests.json');require(len(rows)==2 and all(len(row['raw']['ids'])==524 for row in rows),'Two fresh long524 requests required');views=actor.contract.requests(rows,recipe['requests']);require(canonical(views[0])==canonical(views[1]),'Warmup/selected output repeats differ');arms.append(views)
  require(not list((directory/'host-traces').glob('*')),'H36 disabled but emitted trace');observations.append({'phase':phase,'requests':[{'ordinal':row['ordinal'],'native_DONE':row['raw']['done'],'root_stdout_seconds':row['raw']['stdout_seconds'],'output_ids':row['raw']['output_ids']}for row in rows]})
 left=c.read(root/actor.PHASES[0]/'recipe.json');right=c.read(root/actor.PHASES[1]/'recipe.json');le=dict(left['native_env']);re=dict(right['native_env']);require(le.pop('STRATA_PF_GROUPED_F16')=='0'and re.pop('STRATA_PF_GROUPED_F16')=='1'and canonical(le)==canonical(re)and left['native_argv']==right['native_argv']and canonical(left['requests'])==canonical(right['requests']),'Only groupedF16 switch may differ between fresh45 processes');require(canonical(arms[0])==canonical(arms[1]),'Actual47 OFF/ON complete bounded outputs differ')
 proof=actor.fresh.binding();require(actor.closure()==report['source_plan_sha256']and {str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'}==report['artifacts'],'Closing actual source/raw artifacts changed')
 result.update(source_generation=47,actual47_SDK=proof,original43_runtime_proof_transferred=False,grouped_F16_OFF_ON_outputs_equal=True,native_observations=observations,numeric_first49_observed=False,H36_coverage_qualified=False,graph_coverage_qualified=False,device_coverage_qualified=False,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
