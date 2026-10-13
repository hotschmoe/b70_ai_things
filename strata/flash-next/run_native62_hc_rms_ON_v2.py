"""Fresh combined62 mode0 HC arithmetic OFF/ON; original owned NUM33 boundary."""
import ast,copy,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_source59_mode0_fidelity_v3 as prior
import run_natural_p2048_native59_v1 as natural
import native_source41_cli_protocol_v4 as pipe
import source41_lpdisabled_contract_v1 as omission
import current_mode0_fidelity_capture_recipe_v2 as captured
from serial37_canonical_json_v3 import canonical
c=prior.c;require=prior.require;ROOT=prior.ROOT;HERE=prior.HERE
PLAN=HERE/'native62-hc-rms-ON-capture-source-plan-v2.json'
PHASES=('native62-hc-rms-ON',)
_OFF=None
# Own natural wrapper namespace: no frozen59 module/global mutation.
parent=types.ModuleType('_private_native62_fidelity_parent');parent.__dict__.update(vars(natural));P=parent.__dict__
NS=dict(prior.NS)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is prior.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
for name,value in list(P.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is vars(natural):P[name]=types.FunctionType(value.__code__,P,value.__name__,value.__defaults__,value.__closure__)
P.update(NS=NS,PLAN=PLAN,PHASES=PHASES,SETTINGS=None,fresh=None)
P['BASE_COMMAND']=types.FunctionType(natural.BASE_COMMAND.__code__,NS,natural.BASE_COMMAND.__name__,natural.BASE_COMMAND.__defaults__,natural.BASE_COMMAND.__closure__)
# Only the two explicit generation predicates and CLI generation choices port.
for name in('configure','main'):
 node=copy.deepcopy(next(n for n in ast.parse(Path(natural.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name));count=0
 for n in ast.walk(node):
  if isinstance(n,ast.Constant)and type(n.value)is int and n.value==59:n.value=62;count+=1
 require(count==(2 if name=='configure'else 1),'Exact natural generation derivation count differs')
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[actual62-generation-only]','exec'),P)

def closure():
 require(_OFF is not None and c.sha(_OFF[0])==_OFF[1],'AuthenticatedOFF component byte binding required');component=c.read(_OFF[0]);require(component['passed']is True and component['original_producer_passed']is False and component['producer_status_rewritten']is False and component['original_producer_report_sha256']=='72e1867f56f2748483f1529cb33ff00c9075bf7069c6cfa0c50cda796bc201d5'and component['explicit_sqrt_divide']==0 and component['baseline_original59']['all33_and49_and144_phase_tensors_bitwise']is True,'Exact authenticated priorOFF scope required')
 natural.base.closure();before=c.sha(PLAN)
 for n,h in c.read(PLAN)['files'].items():require(c.sha(ROOT/n)==h,'Actual62 RMS source changed '+n)
 require(parent.SETTINGS is not None and c.sha(parent.SETTINGS['corpus'])==parent.SETTINGS['corpus_sha256'],'Prepared actual corpus changed')
 if parent.fresh is not None and parent.fresh.SDK is not None:
  require(parent.fresh.PLAN==HERE/'native62-HC-perf-explicit-sqrt-engine-proposal-v1.json','Exact combined62 proposal required')
  for phase in PHASES:parent.arm_profile(phase)
 require(c.sha(PLAN)==before,'Actual62 source plan changed');return before

_recipe_node=copy.deepcopy(next(n for n in ast.parse(Path(captured.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='recipe'));_allow=0
for n in ast.walk(_recipe_node):
 if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==[52,58,59]:n.elts.append(ast.Constant(62));_allow+=1
require(_allow==1,'Exactly current recipe generation whitelist addition required')
_R={**vars(captured)};exec(compile(ast.fix_missing_locations(ast.Module([_recipe_node],type_ignores=[])),__file__+'[existingNUM33-generation62]','exec'),_R)
def phase_recipe(phase,*args):
 require(phase in PHASES,'Exact combined62 arithmetic phase required');m,r=parent.phase_recipe(phase,*args);r.update(_R['recipe'](r['native_argv'],r['native_env'],args[-1],62));r['native_env'].update(STRATA_COPY_ENGINE='0',STRATA_HC_Q8_TOKEN_TILE='2',STRATA_HC_PREFILL_COALESCE='1',STRATA_HC_PREFILL_TILE4='1',STRATA_HC_EXPLICIT_SQRT_DIVIDE='1')
 r.update(mode=phase,NUM33_enabled=True,PTI_enabled=False,tool_runtime=NS['_RUNTIME'],engine_generation=62,selected_request_count=4,maximum_total_output_tokens=4,explicit_sqrt_divide=1,matched_experiment='HC-explicit-sqrt-divide',capture_kind='current62 mode0 four literalprefix1 NUM33/all48 P30/SFD49/head; no fullmath gate',runtime_lifecycle_integrated=True,launch_authorized=True,source_only_recipe=False)
 m['alias']='native62-mode0-HC-explicit-sqrt-divide-NUM33-native-CLI';return m,r
class Protocol(pipe.Protocol):
 def __init__(self,command,directory,*args,**kwargs):
  self.roster=c.read(Path(directory)/'recipe.json')['requests'];(Path(directory)/'layer0').mkdir();kwargs.update(pipe_bytes=128<<20,pipe_lines=1000000);super().__init__(command,directory,*args,**kwargs);(self.directory/'ARM').write_text('ARM actual62 literalprefix1 NUM33 P30\n',encoding='ascii')
 def request(self,label,ids,**kwargs):
  i=int(label.removeprefix('request-'))-1;want=self.roster[i];require(ids==want['ids']==[248045]and kwargs.get('fresh')==1 and kwargs.get('max_new')==1 and kwargs.get('pin')is None,'Exact fresh literal1/MAX1 required');r=omission._REQUEST['request'](self,label,ids,**kwargs);r['capture_directory']=str((self.directory/'captures').resolve());return r

def requests(rows,expected):
 values=omission.requests(rows,expected);require(len(values)==4 and all(r['ids']==[248045]for r in values),'Four fresh literal requests required');require(all(canonical(x)==canonical(values[0])for x in values),'Within-arm first-row repeats differ');return values

def equivalence(*arms):
 require(len(arms)==1 and len(arms[0])==4 and all(canonical(x)==canonical(arms[0][0])for x in arms[0]),'Four ON repeats must agree');return {'ON_repeat_coherent':True,'OFF_component_reused':True,'numeric_pass_assigned':False,'full_model_math_qualified':False}

def matched_recipe_gate(left,right):
 require(left['native_argv']==right['native_argv']and canonical(left['requests'])==canonical(right['requests'])and canonical(left['expert_profile_binding'])==canonical(right['expert_profile_binding']),'Matched actual source/config/input/profile required');a=dict(left['native_env']);b=dict(right['native_env']);require(a.pop('STRATA_HC_EXPLICIT_SQRT_DIVIDE')=='0'and b.pop('STRATA_HC_EXPLICIT_SQRT_DIVIDE')=='1'and canonical(a)==canonical(b),'Only explicit sqrt/divide flag may differ');return True

def native_command(*args):return parent.native_command(*args)
def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);source=closure();commands=[]
 for phase in PHASES:
  desc,r=phase_recipe(phase,m,cfg,inputs,source);require(r['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']in('0','1')and r['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0','Cheap actual flags required before payload/lease');name,cmd=native_command(Path(output)/phase,desc,r,NS['_CASE'][1],source);commands.append({'phase':phase,'container_name':name,'native_command':cmd,'recipe':r})
 require(len(commands)==1 and commands[0]['recipe']['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']=='1','Single actualON recipe required');return {'pure_command_construction_passed':True,'GPU_launch_performed':False,'commands':commands,'newSDK_actual_ABI_admission_still_required':True}

BASE_RUN_PHASE=types.FunctionType(prior.BASE_RUN_PHASE.__code__,NS,prior.BASE_RUN_PHASE.__name__,prior.BASE_RUN_PHASE.__defaults__,prior.BASE_RUN_PHASE.__closure__)
def run_phase(*args,**kwargs):
 result=BASE_RUN_PHASE(*args,**kwargs)
 from native62_hc_rms_raw_v2 import collect
 out,phase=args[:2];directory=out/phase;raw=collect(directory,args[5],sys.modules[__name__]);c.write(directory/'fidelity-raw.json',raw)
 from native62_hc_rms_comparison_v2 import independent_metrics
 comparison=independent_metrics(raw);c.write(directory/'numerical-comparison.json',comparison);c.write(directory/'OFF-component.snapshot.json',c.read(_OFF[0]));c.write(directory/'OFF-component-binding.json',{'path':str(_OFF[0]),'sha256':_OFF[1]})
 from source41_pti_raw_budget_v2 import publish_final
 require(publish_final(directory/'raw-budget.json',c.read(directory/'raw-budget.json'),out)['passed'],'Final actual62 bounded raw budget failed');return result
NS.update(__file__=__file__,PLAN=PLAN,PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight,run_phase=run_phase,matched_recipe_gate=matched_recipe_gate)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})
from source59_trace_pair_startup_v1 import memory_admission
NS['life']=SimpleNamespace(**{**vars(NS['life']),'prepared_ctrl':SimpleNamespace(**{**vars(NS['life'].prepared_ctrl),'memory_admission':memory_admission}),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
P.update(closure=closure,preflight=preflight,NS=NS,PHASES=PHASES,set_case=NS['set_case'],set_runtime=NS['set_runtime'])
NS['set_sdk']=parent.set_sdk
for n in('contract','set_case','set_runtime','admission','prerequisite','life','warm_control','mapped','native_pid','stat_snapshot','campaign_stats'):globals()[n]=NS[n]
def main():
 global _OFF
 import argparse
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--off-component',type=Path,required=True);parser.add_argument('--off-component-sha256',required=True);options,rest=parser.parse_known_args(sys.argv[1:]);_OFF=(options.off_component.resolve(),options.off_component_sha256);sys.argv=[sys.argv[0],*rest]
 require(__debug__ and sys.flags.optimize==0 and not os.environ.get('PYTHONOPTIMIZE'),'Unoptimized actual62 capture required');rest=sys.argv[1:];require('--generation'in rest and rest[rest.index('--generation')+1]=='62','Only actual combined62 allowed')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*rest,'--off-component',str(_OFF[0]),'--off-component-sha256',_OFF[1],'--leased'])
 NS['_PORTS']=parent.base.boundary_ports(NS);parent.main()
if __name__=='__main__':main()
