#!/usr/bin/env python3
"""ON-only retry; completed failed-parent OFF component remains separate."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import run_source43_warm_pti_v3 as original
import pti_warm_owned_exec_v3 as temporal_exec
import validate_source43_warm_off_component_v1 as component
from serial37_canonical_json_v3 import canonical

def set_baseline(root,report_sha):
 value=component.finalized_binding(root,report_sha)
 original.require(value['historical_metadata_only']is True and value['current_model_identity_qualified']is False and value['current_GPU_health_transferred']is False and value['fresh_public_full4_rehashed']is False,'Historical baseline cannot confer current model/health/hash authority')
 NS['_BASELINE']=value

def baseline_snapshot():return NS['_BASELINE']

def baseline_equivalence(*arms):
 original.require(len(arms)==1 and NS['_BASELINE']is not None,'One actual ON arm and original OFF component required')
 original.require(canonical(arms[0])==canonical(NS['_BASELINE']['request_views']),'Actual ON IDs/finish differ from original closed OFF component')
 return {'H36_OFF_ON_equivalence_passed':True,'scope':'new ON two fresh GEN4 rows vs original closed OFF component; failed baseline parent remains failed'}

def preflight(plan,output):
 m=original.c.read(Path(plan['template_path'])/'prepared.json');cfg=original.c.read(Path(plan['template_path'])/'server-config.json');inputs=original.c.read(plan['inputs_path'])
 original.require(original.c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and original.c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256'],'Exact small template bytes required before current admission')
 descriptor,recipe=NS['phase_recipe'](NS['PHASES'][0],m,cfg,inputs,original.c.sha(NS['PLAN']))
 name,command=NS['life'].contract.native_command(Path(output)/NS['PHASES'][0],descriptor,recipe,NS['_CASE'][1],original.c.sha(NS['PLAN']))
 original.require(NS['life'].contract.native_command.__globals__ is NS and NS['_RUNTIME']['install_root']+':/pti:ro'in command,'Actual private runtime/command builder must bind before admission')
 return {'pure_command_construction_passed':True,'native_command':command,'container_name':name,'current_admission_still_required':True,'GPU_launch_performed':False}

def derive():
 ns={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source43-warm-pti-source-plan-v5.json','PHASES':('source43-warm-pti-ON',),'temporal_exec':temporal_exec,'set_baseline':set_baseline,'baseline_snapshot':baseline_snapshot,'preflight':preflight,'_BASELINE':None}
 ns['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':baseline_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(True);count+=1
 original.require(count==1,'Sole actual ON predicate required')
 class Delta(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source43_warm_pti_v5'
   return n
  def visit_Constant(self,n):
   if type(n.value)is str and "p.add_argument('--pti-intake'"in n.value:
    n.value=n.value.replace("    a = p.parse_args()","    p.add_argument('--baseline-root',type=Path,required=True)\n    p.add_argument('--baseline-report-sha256',required=True)\n    a = p.parse_args()\n    set_baseline(a.baseline_root,a.baseline_report_sha256)")
   if type(n.value)is str and "report['OFF_prerequisite']=prior"in n.value:
    n.value=n.value.replace("    phases = PHASES","    report['OFF_component_prerequisite']=baseline_snapshot()\n    report['command_preflight']=command_preflight\n    phases = PHASES")
   return n
 port=next(n for n in nodes if n.name=='ports');insert=next(i for i,n in enumerate(port.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
 port.body[insert:insert]=ast.parse("require(source.count('metadata, cfg = admission(plan)')==1,'Exact current admission call required')").body+[ast.Assign(targets=[ast.Name(id='source',ctx=ast.Store())],value=ast.Call(func=ast.Attribute(value=ast.Name(id='source',ctx=ast.Load()),attr='replace',ctx=ast.Load()),args=[ast.Constant('metadata, cfg = admission(plan)'),ast.Constant('command_preflight = preflight(plan, a.output)\n    metadata, cfg = admission(plan)')],keywords=[]))]

 nodes=[Delta().visit(n)for n in nodes];exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[ON-only-original-protected-parent]','exec'),ns);ns['life']=SimpleNamespace(**{**vars(original.life),'contract':SimpleNamespace(**{**vars(original.life.contract),'native_command':ns['native_command']})});ns['_PORTS']=ns['ports']();return ns
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):return NS[name]
if __name__=='__main__':NS['_PORTS']['main']()
