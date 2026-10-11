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

_API=None;_API_SHA=None

def set_api(root,expected_sha):
 global _API,_API_SHA
 root=Path(root);original.require(root.is_absolute()and root.resolve()==root and original.c.sha(root/'prepared.json')==expected_sha,'Exact current174 prepared metadata bytes required');_API=root;_API_SHA=expected_sha

def closure():
 plan=original.HERE/'source43-expert-lanes32-source-plan-v1.json'
 for name,want in original.c.read(plan)['files'].items():original.require(original.c.sha(original.ROOT/name)==want,'Exact shortread2 source changed '+name)
 import c1_source43_api_controller_v3 as api
 api.source_binding();registry=api.registry_binding();original.require(registry['actual_models']==174 and registry['all173_complete_rows_preserved']is True and registry['old173_current_global_hash_gate_passed']is False,'Exact174 association preserving all173 required');return original.c.sha(plan)

def prerequisite(root):
 original.require(_API is not None,'Explicit current174 API prepared source required');m=original.c.read(_API/'prepared.json');normal=m['historical_prerequisites']['normal'];original.require(Path(root).resolve()==Path(normal['root']).resolve()and original.c.sha(_API/'prepared.json')==_API_SHA,'Only the authenticated named normal history is allowed')
 return {'scope':'original normal metadata association; full current API admission still required','root':normal['root'],'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'API_prepared_root':str(_API),'API_prepared_sha256':_API_SHA}

def admission(plan):
 import c1_source43_api_controller_v3 as api
 original.require(_API is not None and original.c.sha(_API/'prepared.json')==_API_SHA,'Current prepared association changed');current=api.validate_prepared(_API);root=Path(current['historical_prerequisites']['normal']['root']);m=original.c.read(root/'prepared.snapshot.json');cfg=original.c.read(root/'config.snapshot.json')
 original.require(canonical(cfg)==canonical(current['original_template_config'])and canonical(plan['source43'])==canonical(current['source43_sdk']),'Actual current43 SDK/config association differs')
 original.require(canonical(original.c.read(Path(plan['template_path'])/'prepared.json'))==canonical(m)and original.c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and original.c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256']and canonical(original.c.read(Path(plan['template_path'])/'server-config.json'))==canonical(cfg),'Original CLI template bytes/config differ from authenticated normal snapshots')
 original.require(m['pack']==current['pack']and canonical(m['runtime'])==canonical(current['runtime'])and m['runtime_receipt']==current['runtime_receipt'],'Current actual43 runtime/pack differs')
 original.require(original.c.sha(plan['actual_model_identity_path'])==plan['actual_model_identity_sha256']and original.c.sha(plan['inputs_path'])==plan['inputs_sha256'],'Actual model identity/input receipt changed');return m,cfg

def preflight(plan,output):
 m=original.c.read(Path(plan['template_path'])/'prepared.json');cfg=original.c.read(Path(plan['template_path'])/'server-config.json');inputs=original.c.read(plan['inputs_path'])
 original.require(original.c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and original.c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256'],'Small actual template byte pins required')
 descriptor,recipe=NS['phase_recipe'](NS['PHASES'][0],m,cfg,inputs,original.c.sha(NS['PLAN']));name,command=NS['life'].contract.native_command(Path(output)/NS['PHASES'][0],descriptor,recipe,NS['_CASE'][1],original.c.sha(NS['PLAN']))
 original.require(NS['life'].contract.native_command.__globals__ is NS and recipe['PTI_enabled']is False and recipe['native_argv'][-2:]==['--short-read','0']and len(recipe['requests'])==2 and all(recipe['native_env'][k]=='32'for k in('STRATA_GU_LANES','STRATA_DOWN_LANES')),'Actual private single-OFF shortread0 lanes32 command required');return {'pure_command_construction_passed':True,'native_command':command,'container_name':name,'GPU_launch_performed':False,'current_admission_still_required':True}

def derive():
 namespace={**vars(original),'__file__':__file__,'PLAN':original.HERE/'source43-expert-lanes32-source-plan-v1.json','PHASES':('source43-warm-pti-OFF',),'closure':closure,'admission':admission,'prerequisite':prerequisite,'preflight':preflight,'set_api':set_api}
 namespace['contract']=SimpleNamespace(**{**vars(original.contract),'equivalence':repeat_equivalence})
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)]
 phase=next(n for n in nodes if n.name=='phase_recipe');count=0
 for n in ast.walk(phase):
  if isinstance(n,ast.keyword)and n.arg=='PTI_enabled':n.value=ast.Constant(False);count+=1
 original.require(count==1,'Sole native OFF phase predicate required')
 phase.body[-1:-1]=ast.parse("require('--short-read'not in recipe['native_argv'],'Original default short-read geometry required');recipe['native_argv'].extend(['--short-read','0']);recipe.update(short_read_control=0,only_native_argument_delta=['--short-read','0'],numeric_head_residual_observed=False)").body
 phase.body[-1:-1]=ast.parse("require(all(recipe['native_env'].get(k)in(None,'8')for k in('STRATA_GU_LANES','STRATA_DOWN_LANES')),'Baseline dispatch must be default8');recipe['native_env'].update(STRATA_GU_LANES='32',STRATA_DOWN_LANES='32');recipe.update(expert_lane_control={'GU':32,'DOWN':32},combined_dispatch_factor=True)").body
 # The run-phase's unchanged graph proof executes with the actual new reader.
 class ReaderName(ast.NodeTransformer):
  def visit_ImportFrom(self,n):
   if n.module=='validate_source43_warm_pti_v3':n.module='validate_source43_expert_lanes32_v1'
   return n
 nodes=[ReaderName().visit(n)for n in nodes if n.name not in('closure','admission','prerequisite')]
 port=next(n for n in nodes if n.name=='ports')
 class ParentDelta(ast.NodeTransformer):
  def visit_Constant(self,n):
   if type(n.value)is str and "p.add_argument('--pti-intake'"in n.value:n.value=n.value.replace("    a = p.parse_args()","    p.add_argument('--api-prepared',type=Path,required=True)\n    p.add_argument('--api-prepared-sha256',required=True)\n    a = p.parse_args()\n    set_api(a.api_prepared,a.api_prepared_sha256)")
   if type(n.value)is str and "report['OFF_prerequisite']=prior"in n.value:n.value=n.value.replace("    phases = PHASES","    report['API_prepared_root']=str(a.api_prepared)\n    report['API_prepared_sha256']=a.api_prepared_sha256\n    report['command_preflight']=command_preflight\n    phases = PHASES")
   return n
 nodes=[ParentDelta().visit(n)for n in nodes]
 insert=next(i for i,n in enumerate(port.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
 port.body[insert:insert]=ast.parse("source=source.replace('metadata, cfg = admission(plan)','command_preflight = preflight(plan, a.output)\\n    metadata, cfg = admission(plan)')").body
 exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[single-OFF-short-read0]', 'exec'),namespace)
 namespace['life']=SimpleNamespace(**{**vars(original.life),'contract':SimpleNamespace(**{**vars(original.life.contract),'native_command':namespace['native_command']})});namespace['_PORTS']=namespace['ports']();return namespace
NS=derive()
for name in('PLAN','PHASES','contract','closure','set_case','set_runtime','admission','prerequisite','phase_recipe','native_command','life','warm_control','mapped','native_pid','run_phase','ports'):
 globals()[name]=NS[name]
def __getattr__(name):
 if name in NS:return NS[name]
 raise AttributeError(name)
if __name__=='__main__':NS['_PORTS']['main']()
