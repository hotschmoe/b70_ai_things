"""Saved experiment identity reader; safe47 OFF-only construction by default.
Reusable boundary port for future matched controls after actual SDK admission.
No extra whole-model reread, original full-math or speed authority.
"""
import ast,copy
from pathlib import Path
import run_source47_grouped_f16_model_v2 as actor
import validate_source43_bounded_pti_v1 as original
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

def boundary_reader_ports(active_actor=actor):
 import builtins
 ns={**vars(original),'actor':active_actor};captured=[]
 def compiler(source,*args,**kwargs):captured.append(source);return builtins.compile(source,*args,**kwargs)
 ns['compile']=compiler
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='ports'))
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[private-reader-port]','exec'),ns)
 result=ns['ports']();tree=ast.parse(captured[-1]);final=tree.body[0]
 loop=next(n for n in final.body if isinstance(n,ast.For)and ast.unparse(n.target)=='phase')
 # Keep existing source/config/command/health/journal/EOF/retirement/raw predicates.
 loop.body=[n for n in loop.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='full4')]
 for n in ast.walk(loop):
  if isinstance(n,ast.Constant)and n.value=='pre_after_pages':n.value='pre_pages'
 loop.body[:0]=ast.parse("identity=report['per_arm'][phase]\nrequire(canonical(identity['pre_source_binding'])==canonical(a.fresh.binding())and canonical(identity['post_source_binding'])==canonical(a.fresh.binding()),'Per-arm current source/binary binding differs')\npages(identity['pre_pages'],Path(report['pre_full4']['rows'][2]['path']));pages(identity['post_pages'],Path(report['post_full4']['rows'][2]['path']))").body
 for n in list(final.body):
  text=ast.unparse(n)
  if 'aggregate_pre_from_actual_arm' in text or 'Aggregate identity must be exact first/last real arm receipt' in text:final.body.remove(n)
 final.body.insert(3,ast.parse("a.campaign_stats(report,a.stat_snapshot([Path(row['path'])for row in report['pre_full4']['rows']]))").body[0])
 # Final campaign post scan is bracketed on both sides, not merely post-pages.
 at=next(i for i,n in enumerate(final.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='full4')
 final.body[at:at]=ast.parse("pages(report['post_before_pages'],Path(report['post_full4']['rows'][2]['path']))\nrequire(report['post_health']['finished_epoch']<=report['post_before_pages']['epoch']<=report['post_full4']['started'],'Campaign post-health/page/hash chronology differs')").body
 # No extra public full-model scan: inspect the authenticated saved experiment.
 exec(compile(ast.fix_missing_locations(tree),__file__+'[campaign-reader]','exec'),result)
 result['finalized_binding']=result['_finalized_binding']
 return result

def finalized_binding(root):
 root=Path(root);report=c.read(root/'report.json');actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=c.read(root/actor.PHASES[0]/'recipe.json')['tool_runtime'];actor.set_runtime(saved['intake_path'],saved['exact_intake_sha256'])
 result=boundary_reader_ports()['finalized_binding'](root)
 arms=[];observations=[]
 for index,phase in enumerate(actor.PHASES):
  directory=root/phase;recipe=c.read(directory/'recipe.json');require(recipe['engine_generation']==47 and recipe['materialized_mode']==0 and recipe['native_env']['STRATA_PF_GROUPED_F16']=='0'and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['native_argv'][recipe['native_argv'].index('--prefill')+1]=='64','Exact safe native47 groupedOFF and traceOFF required');rows=c.read(directory/'requests.json');require(len(rows)==2 and all(len(row['raw']['ids'])==524 for row in rows),'Two fresh long524 requests required');views=actor.contract.requests(rows,recipe['requests']);require(canonical(views[0])==canonical(views[1]),'Warmup/selected output repeats differ');arms.append(views)
  require(not list((directory/'host-traces').glob('*')),'H36 disabled but emitted trace');observations.append({'phase':phase,'requests':[{'ordinal':row['ordinal'],'native_DONE':row['raw']['done'],'root_stdout_seconds':row['raw']['stdout_seconds'],'output_ids':row['raw']['output_ids']}for row in rows]})
 if len(actor.PHASES)==2:
  left=c.read(root/actor.PHASES[0]/'recipe.json');right=c.read(root/actor.PHASES[1]/'recipe.json');require(canonical(left['native_env'])==canonical(right['native_env'])and left['native_argv']==right['native_argv'] and canonical(left['requests'])==canonical(right['requests']),'Safe baseline matched configuration differs');require(canonical(arms[0])==canonical(arms[1]),'Matched bounded outputs differ')
 proof=actor.fresh.binding();require(actor.closure()==report['source_plan_sha256']and {str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'}==report['artifacts'],'Closing actual source/raw artifacts changed')
 result.update(H36_OFF_ON_equivalence_passed=False,identity_scope='whole_matched_arms',current_public_full4_rehashed=False,saved_campaign_full4_validated=True,source_generation=47,actual47_SDK=proof,original43_runtime_proof_transferred=False,grouped_F16_enabled=False,matched_arms_outputs_equal=len(arms)==2,native_observations=observations,numeric_first49_observed=False,H36_coverage_qualified=False,graph_coverage_qualified=False,device_coverage_qualified=False,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
