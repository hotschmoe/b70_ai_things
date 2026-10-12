"""Read-only scoped ArmA admission; original V5 aggregate failure stays failed."""
import argparse,ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import validate_source52_p2048_profile_v5 as common
import validate_source47_grouped_f16_model_v2 as boundary
from serial37_canonical_json_v3 import canonical
actor=common.actor;c=common.c;require=common.require;HERE=common.HERE;ROOT=common.ROOT
REPORT_SHA='f4c69e0d08e7c6d23a04683593c8c5a15767c662bdca6e130685e7648c357b5f'
ERRORS=['ValueError: Actual native phase failed natural-p2048-B','Postwrite total raw budget exceeded']
PLAN=HERE/'source52-v5-armA-component-reader-source-plan-v1.json'
def source_binding():
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Scoped component reader source changed '+name)
 return c.sha(PLAN)
def component_ports(active):
 # Capture the already-derived boundary reader AST, changing ONLY the named
 # whole-success admission and absent two-arm comparison into explicit scope.
 import builtins,types
 captured=[]
 def compiler(source,*args,**kwargs):
  if isinstance(source,ast.Module)and any(isinstance(n,ast.FunctionDef)and n.name=='_finalized_binding'for n in source.body):captured.append(copy.deepcopy(source))
  return builtins.compile(source,*args,**kwargs)
 fn=boundary.boundary_reader_ports;private=types.FunctionType(fn.__code__,{**vars(boundary),'compile':compiler},fn.__name__,fn.__defaults__,fn.__closure__);ports=private(active);require(captured,'Exact original boundary AST required');tree=captured[-1];node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_finalized_binding');changed=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Call)and getattr(n.func,'id',None)=='require'and len(n.args)==2 and isinstance(n.args[1],ast.Constant):
   if n.args[1].value=='Actual model CLI failed/untyped':
    n.args[0]=ast.parse("report['passed'] is False and report['errors']==EXPECTED_ERRORS and type(report['producer_pid']) is int and report['producer_pid']>0",mode='eval').body;n.args[1]=ast.Constant('Exact original failed V5 aggregate scope required');changed.append('named_failed_parent')
   elif n.args[1].value=='Actual OFF comparison differs':
    n.args[0]=ast.parse("len(arms)==1 and len(arms[0])==2 and canonical(arms[0][0])==canonical(arms[0][1])",mode='eval').body;n.args[1]=ast.Constant('Scoped actual A warmup/selected output differs');changed.append('one_arm_comparison')
 require(sorted(changed)==['named_failed_parent','one_arm_comparison'],'Only two explicit component-scope predicate changes');ports['EXPECTED_ERRORS']=ERRORS;exec(compile(ast.fix_missing_locations(tree),__file__+'[unchanged-boundary-except-declared-component-scope]','exec'),ports);return ports['_finalized_binding']
def component_budget(directory):
 v=c.read(directory/'raw-budget.json');require(v['passed']is True and v['errors']==[]and v['thread_retired']is True and v['cap_bytes']==1<<30 and v['interval_seconds']==1,'Original ArmA whole-root watcher passed before B required');peak=0;last=0
 for row in v['samples']:
  require(type(row['bytes'])is int and row['bytes']==sum(x['bytes']for x in row['files'])and row['monotonic_ns']>=last,'Original ArmA watcher sample differs');last=row['monotonic_ns'];peak=max(peak,row['bytes'])
 require(peak==v['peak_observed_bytes']<=1<<30 and v['observed_overshoot_bytes']==0 and v['actual_postwrite_total_bytes']<=1<<30 and v['observed_postwrite_overshoot_bytes']==0,'Original ArmA observation/close exceeded original budget');size=sum(p.stat().st_size for p in directory.rglob('*')if p.is_file());require(size<=1<<30,'Current immutable A component exceeds original individual limit');return {'original_watcher_scope':'whole original root at A completion, before B','original_passed':True,'observed_peak_bytes':peak,'original_postwrite_total_bytes':v['actual_postwrite_total_bytes'],'current_component_bytes':size,'original_experiment_aggregate_gate_waived':False}
def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');require(c.sha(root/'report.json')==REPORT_SHA and report['passed']is False and report['errors']==ERRORS,'Only exact known failed V5 parent admitted for scoped reader');before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Exact failed parent full artifact tree differs');phase='natural-p2048-A';directory=root/phase;recipe=c.read(directory/'recipe.json')
 try:
  with common.configured_actor(report,recipe)as active:
   scoped=SimpleNamespace(**{**vars(active),'PHASES':(phase,)})
   result=component_ports(scoped)(root);budget=component_budget(directory)
   # Reuse original arm predicates, except its current-whole-root budget call:
   # B made that ORIGINAL aggregate fail. Current component budget is explicit.
   fn=copy.deepcopy(next(n for n in ast.parse(Path(common.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='arm_binding'));target=[n for n in ast.walk(fn)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='pti_reader.old.budget'];require(len(target)==1,'Exact original whole-root budget call required');target[0].value=ast.Call(func=ast.Name(id='component_budget',ctx=ast.Load()),args=[ast.Name(id='directory',ctx=ast.Load())],keywords=[]);ns={**vars(common),'component_budget':component_budget};exec(compile(ast.fix_missing_locations(ast.Module([fn],type_ignores=[])),__file__+'[declared-A-component-budget]','exec'),ns);arm=ns['arm_binding'](root,report,directory,recipe,active);require(arm['copy_engine']=='0','Only genuine complete original COPY0 A component required');result.update(component_admitted=True,producer_admitted=False,original_parent_passed=False,original_parent_report_sha256=REPORT_SHA,original_parent_errors=ERRORS,original_aggregate_budget_failed=True,component='natural-p2048-A',copy_engine=0,arm=arm,component_budget=budget,actual52_SDK=active.fresh.binding(),reader_source_plan_sha256=source,matched_trace_arms_qualified=False,H36_OFF_ON_equivalence_passed=False,H36_coverage_qualified=False,Chrome_CPU_host_lane_recorded=False,clock_alignment_qualified=False,complete_model_operation_coverage_qualified=False,physical_bus_bandwidth_qualified=False,clean_serving_speed_qualified=False,inspector_component_execution_allowed=True);return result
 finally:require(source_binding()==source and c.sha(root/'report.json')==REPORT_SHA and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing original failed parent/source/tree changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external scoped proof required');c.write(a.output,finalized_binding(a.root))
