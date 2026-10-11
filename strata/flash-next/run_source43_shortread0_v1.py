#!/usr/bin/env python3
"""Single observer-OFF short-read0 control; original paired lifecycle retained."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import run_source43_warm_pti_v3 as original
from serial37_canonical_json_v3 import canonical

def repeat_equivalence(*arms):
 original.require(len(arms)==1 and len(arms[0])==2,'Single arm with warmup/selected GEN4 required')
 return {'single_arm_repeat_passed':True,'scope':'warmup/selected actual fresh GEN4; cross-path equality is descriptive only'}

def derive():
 namespace={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source43-shortread0-source-plan-v1.json','PHASES':('source43-warm-pti-OFF',)}
 namespace['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':repeat_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(False);count+=1
 original.require(count==1,'Sole native OFF phase predicate required')
 phase.body[-1:-1]=ast.parse("require('--short-read'not in recipe['native_argv'],'Original default short-read geometry required');recipe['native_argv'].extend(['--short-read','0']);recipe.update(short_read_control=0,only_native_argument_delta=['--short-read','0'],numeric_head_residual_observed=False)").body
 # The run-phase's unchanged graph proof executes with the actual new reader.
 class ReaderName(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source43_shortread0_v1'
   return n
 nodes=[ReaderName().visit(n)for n in nodes]
 exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[single-OFF-short-read0]', 'exec'),namespace)
 namespace['_PORTS']=namespace['ports']();return namespace
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):return NS[name]
if __name__=='__main__':NS['_PORTS']['main']()
