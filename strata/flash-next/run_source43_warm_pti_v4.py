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

def derive():
 ns={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source43-warm-pti-source-plan-v4.json','PHASES':('source43-warm-pti-ON',),'temporal_exec':temporal_exec,'set_baseline':set_baseline,'baseline_snapshot':baseline_snapshot,'_BASELINE':None}
 ns['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':baseline_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(True);count+=1
 original.require(count==1,'Sole actual ON predicate required')
 class Delta(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source43_warm_pti_v4'
   return n
  def visit_Constant(self,n):
   if type(n.value)is str and "p.add_argument('--pti-intake'"in n.value:
    n.value=n.value.replace("    a = p.parse_args()","    p.add_argument('--baseline-root',type=Path,required=True)\n    p.add_argument('--baseline-report-sha256',required=True)\n    a = p.parse_args()\n    set_baseline(a.baseline_root,a.baseline_report_sha256)")
   if type(n.value)is str and "report['OFF_prerequisite']=prior"in n.value:
    n.value=n.value.replace("    phases = PHASES","    report['OFF_component_prerequisite']=baseline_snapshot()\n    phases = PHASES")
   return n
 nodes=[Delta().visit(n)for n in nodes];exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[ON-only-original-protected-parent]','exec'),ns);ns['_PORTS']=ns['ports']();return ns
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):return NS[name]
if __name__=='__main__':NS['_PORTS']['main']()
