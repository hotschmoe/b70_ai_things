"""Actual59 mode0 four fresh NUM33/P30 requests; original owned CLI boundary."""
import argparse,ast,copy,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native59_v1 as parent
import native_source41_cli_protocol_v4 as pipe
import source41_lpdisabled_contract_v1 as omission
import current_mode0_fidelity_capture_recipe_v2 as capture
from serial37_canonical_json_v3 import canonical
c=parent.c;require=parent.require;ROOT=parent.ROOT;HERE=parent.HERE
PLAN=HERE/'source59-mode0-fidelity-capture-source-plan-v2.json'
PHASES=('source59-mode0-fidelity',)
SDK=Path('/mnt/vm_8tb/b70/build/strata-native59-incremental58-v1-20261012T215430Z-tmxlm889')
SDK_SHA='df52f1b799d013e84438dc50e77c4b81b7d5f655fa04ba14c0af05b6d0ea74d0'
def closure():
 parent.closure()
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Current59 fidelity source changed '+name)
 require(parent.fresh is not None and parent.fresh.SDK==SDK and c.sha(SDK/'receipt.json')==SDK_SHA,'Exact actual59 receipt required')
 return c.sha(PLAN)
def phase_recipe(phase,*args):
 require(phase in PHASES,'Single current59 fidelity phase required')
 m,r=parent.phase_recipe(phase,*args);observed=capture.recipe(r['native_argv'],r['native_env'],args[-1],59)
 r.update(observed);r.update(mode=phase,NUM33_enabled=True,PTI_enabled=False,tool_runtime=NS['_RUNTIME'],engine_generation=59,selected_request_count=4,maximum_total_output_tokens=4,capture_kind='current mode0 literalprefix1 NUM33/P30/SFD49 and independent PLE-input33',runtime_lifecycle_integrated=True,launch_authorized=True,source_only_recipe=False)
 m['alias']='source59-mode0-NUM33-P30-native-CLI-diagnostic';return m,r
class Protocol(pipe.Protocol):
 def __init__(self,command,directory,*args,**kwargs):
  self.roster=c.read(Path(directory)/'recipe.json')['requests'];(Path(directory)/'layer0').mkdir()
  kwargs.update(pipe_bytes=128<<20,pipe_lines=1000000)
  super().__init__(command,directory,*args,**kwargs)
  (self.directory/'ARM').write_text('ARM current59 four fresh literalprefix1 NUM33 P30\n',encoding='ascii')
 def request(self,label,ids,**kwargs):
  index=int(label.removeprefix('request-'))-1;want=self.roster[index]
  require(ids==want['ids']==[248045] and kwargs.get('fresh')==1 and kwargs.get('max_new')==1 and kwargs.get('pin')is None,'Exact current59 fresh literal MAX1 required')
  raw=omission._REQUEST['request'](self,label,ids,**kwargs);raw['capture_directory']=str((self.directory/'captures').resolve());return raw

def requests(rows,expected):
 values=omission.requests(rows,expected);require(len(values)==4 and all(r['ids']==[248045] for r in values),'Four literal current59 first rows required')
 require(all(canonical(row)==canonical(values[0])for row in values),'Actual four fresh histories/finish differ');return values

def equivalence(*arms):
 require(len(arms)==1 and len(arms[0])==4 and all(canonical(v)==canonical(arms[0][0])for v in arms[0]),'Actual one-process four-repeat first-row history differs')
 return {'current59_four_fresh_prefix1_histories_equal':True,'full_model_math_qualified':False,'observer_OFF_ON_comparison_performed':False}

def native_command(*args):
 ps["SETTINGS"]=parent.SETTINGS;ps["NS"]=NS
 return PARENT_COMMAND(*args)

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);descriptor,r=phase_recipe(PHASES[0],m,cfg,inputs,closure())
 from native59_startup_option_preflight_v1 import check
 check(r,Path(descriptor['source']));name,command=native_command(Path(output)/PHASES[0],descriptor,r,NS['_CASE'][1],closure())
 require(r['engine_generation']==59 and r['materialized_mode']==0 and r['PTI_enabled']is False and r['cards']==[0,1] and r['stages']==[[0,0,24],[1,24,48]],'Exact current59 paired mode0 required')
 require(len(r['requests'])==4 and all(x['fresh']==1 and x['ids']==[248045]and x['pin']is None and x['max_new']==1 for x in r['requests']),'Exact current59 request roster required')
 return {'pure_command_construction_passed':True,'GPU_launch_performed':False,'commands':[{'phase':PHASES[0],'container_name':name,'native_command':command,'recipe':r}],'current_admission_still_required':True}

NS=dict(parent.NS)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
# This base builder uses the active private runtime; never leave its life callback on the old NS.
ps={**vars(parent),'BASE_COMMAND':types.FunctionType(parent.BASE_COMMAND.__code__,NS,parent.BASE_COMMAND.__name__,parent.BASE_COMMAND.__defaults__,parent.BASE_COMMAND.__closure__)}
PARENT_COMMAND=types.FunctionType(parent.native_command.__code__,ps,parent.native_command.__name__,parent.native_command.__defaults__,parent.native_command.__closure__)
BASE_RUN_PHASE=NS['run_phase']
def run_phase(*args,**kwargs):
 result=BASE_RUN_PHASE(*args,**kwargs)
 from source59_mode0_fidelity_raw_v2 import collect
 out,phase=args[:2];directory=out/phase;raw=collect(directory,args[5]);c.write(directory/'fidelity-raw.json',raw)
 from source41_pti_raw_budget_v2 import publish_final
 proof=publish_final(directory/'raw-budget.json',c.read(directory/'raw-budget.json'),out);require(proof['passed'],'Final current59 raw postwrite budget failed');return result
NS.update(__file__=__file__,PLAN=PLAN,PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight,run_phase=run_phase)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})
NS['life']=SimpleNamespace(**{**vars(NS['life']),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})

def main():
 require(__debug__ and sys.flags.optimize==0 and not os.environ.get('PYTHONOPTIMIZE'),'Unoptimized current59 producer required')
 rest=sys.argv[1:];require('--generation'in rest and rest[rest.index('--generation')+1]=='59','Only current59 allowed')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*rest,'--leased'])
 old_ns,old_phases=parent.NS,parent.PHASES;old_case,old_runtime,old_pre=parent.set_case,parent.set_runtime,parent.preflight
 try:
  parent.NS=NS;parent.PHASES=PHASES;ps['NS']=NS;ps['SETTINGS']=parent.SETTINGS
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  # Profile wrapper consults current parent.SETTINGS after configure; explicit closure avoids stale copy.
  ps['arm_profile']=parent.arm_profile
  NS['_PORTS']=parent.base.boundary_ports(NS)
  parent.main()
 finally:parent.NS=old_ns;parent.PHASES=old_phases;parent.set_case=old_case;parent.set_runtime=old_runtime;parent.preflight=old_pre
if __name__=='__main__':main()
