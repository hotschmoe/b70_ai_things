"""Fresh QSA35 numeric targets; actual successor SDK only, optional diagnostic identity."""
import argparse,ast,copy,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native62_v1 as natural
import native_source41_cli_protocol_v4 as pipe
import source41_lpdisabled_contract_v1 as omission
import current_mode0_fidelity_capture_recipe_v2 as capture
import diagnostic_model_identity_lane_v1 as identity
from serial37_canonical_json_v3 import canonical
c=natural.c;require=natural.require;ROOT=natural.ROOT;HERE=natural.HERE
PLAN=HERE/'source62-qsa35-numeric-capture-source-plan-v1.json';PHASES=('source62-qsa35-first-numeric',);MODE='qualifying';PRIOR=None
parent=types.ModuleType('_private_source62_qsa35_numeric_parent');parent.__dict__.update(vars(natural));P=parent.__dict__;NS=dict(natural.NS)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is natural.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
for name,value in list(P.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is vars(natural):P[name]=types.FunctionType(value.__code__,P,value.__name__,value.__defaults__,value.__closure__)
P.update(NS=NS,PLAN=PLAN,PHASES=PHASES,SETTINGS=None,fresh=None)
P['BASE_COMMAND']=types.FunctionType(natural.BASE_COMMAND.__code__,NS,natural.BASE_COMMAND.__name__,natural.BASE_COMMAND.__defaults__,natural.BASE_COMMAND.__closure__)
node=copy.deepcopy(next(n for n in ast.parse(Path(natural.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='configure'));changed=0
for n in ast.walk(node):
 if isinstance(n,ast.BinOp)and ast.unparse(n)=="HERE / ('source%d_model_sdk_binding_v1.py' % settings['generation'])":n.right=ast.Constant('source62_qsa35_sdk_binding_v2.py');changed+=1
require(changed==1,'Exact single current SDK binder pathname port required');exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[actual-qsa35-binder]','exec'),P)
node=copy.deepcopy(next(n for n in ast.parse(Path(capture.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='recipe'));changed=0
for n in ast.walk(node):
 if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==[52,58,59]:n.elts.append(ast.Constant(62));changed+=1
require(changed==1,'Exact existing recipe generation62 addition');R={**vars(capture)};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[existing-mode0-recipe]','exec'),R)
def closure():
 natural.base.closure();before=c.sha(PLAN)
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Current QSA35 capture source changed '+name)
 require(parent.SETTINGS is not None and c.sha(parent.SETTINGS['corpus'])==parent.SETTINGS['corpus_sha256'],'Original corpus changed')
 if parent.fresh is not None and parent.fresh.SDK is not None:
  require(parent.fresh.PLAN==HERE/'native62-qsa35-session-math-engine-proposal-v1.json','Actual new QSA35 proposal only');parent.arm_profile(PHASES[0])
 require(c.sha(PLAN)==before,'Closing QSA35 source changed');return before

def phase_recipe(phase,*args):
 require(phase==PHASES[0],'One exact QSA35 owner required');m,r=parent.phase_recipe(phase,*args);r.update(R['recipe'](r['native_argv'],r['native_env'],args[-1],62));env=r['native_env'];env.update(STRATA_COPY_ENGINE='0',STRATA_HC_Q8_TOKEN_TILE='2',STRATA_HC_PREFILL_COALESCE='1',STRATA_HC_PREFILL_TILE4='1',STRATA_HC_EXPLICIT_SQRT_DIVIDE='1',STRATA_PF_GDN_DECODER_PROJ='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_QSA3_TARGET='1',STRATA_QSA3_TARGET_DIR='/results/qsa',STRATA_QSA3_TARGET_BINDING_SHA256=args[-1],STRATA_QSA3_TARGET_LAYER='35');r['native_argv'][r['native_argv'].index('--ple-io')+1]='direct';r['native_argv'][r['native_argv'].index('--ple-row-cache')+1]='65536'
 r.update(mode=phase,NUM33_enabled=False,QSA35_enabled=True,QSA3_target_enabled=True,PTI_enabled=False,tool_runtime=NS['_RUNTIME'],engine_generation=62,source_successor='qsa35_session_math_v1',selected_request_count=4,maximum_total_output_tokens=4,required_directories=['captures','p30','ple-input','qsa'],explicit_sqrt_divide=1,diagnostic_identity=(MODE=='diagnostic'),identity_mode=MODE,prior_identity=PRIOR,raw_layout='Original P30 all48/3phases + SFD49/fullhead +PLE input +QSA29 atlayer35/stage1 and stage0zero acknowledgment',capture_kind='Fourfresh literal248045/MAX1 ctx4096 QSA35 numeric comparison targets',runtime_lifecycle_integrated=True,launch_authorized=True,source_only_recipe=False,numeric_pass_assigned=False,full_model_math_qualified=False)
 m['alias']='source62-qsa35-literal1-native-numeric-targets';return m,r
class Protocol(pipe.Protocol):
 def __init__(self,command,directory,*args,**kwargs):
  self.roster=c.read(Path(directory)/'recipe.json')['requests'];(Path(directory)/'qsa').mkdir(exist_ok=True);kwargs.update(pipe_bytes=128<<20,pipe_lines=1000000);super().__init__(command,directory,*args,**kwargs);(self.directory/'ARM').write_text('ARM QSA35 fourfresh literal1 source-bound targets\n',encoding='ascii')
 def request(self,label,ids,**kwargs):
  i=int(label.removeprefix('request-'))-1;want=self.roster[i];require(ids==want['ids']==[248045]and kwargs.get('fresh')==1 and kwargs.get('max_new')==1 and kwargs.get('pin')is None,'Actual fresh literal/MAX1/absentPIN required');return omission._REQUEST['request'](self,label,ids,**kwargs)
def requests(rows,expected):
 views=omission.requests(rows,expected);require(len(views)==4 and all(r['ids']==[248045]for r in views)and all(canonical(x)==canonical(views[0])for x in views),'Four original literal first-output repeats required');return views
def equivalence(*arms):
 require(len(arms)==1 and len(arms[0])==4 and all(canonical(x)==canonical(arms[0][0])for x in arms[0]),'Four current repeats must agree');return dict(QSA35_first_output_repeat_coherent=True,diagnostic_identity=(MODE=='diagnostic'),numeric_pass_assigned=False,full_model_math_qualified=False,speed_qualified=False,stability_qualified=False,shelf_qualified=False)
def native_command(*args):return parent.native_command(*args)
def preflight(plan,output):
 if MODE=='diagnostic':
  lock=c.read(HERE/'model-lock.json');identity.authenticate(c,PRIOR,HERE/'model-lock.json',lock,[ROOT/lock['destination']/x['path']for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')])
 require(parent.SETTINGS['cards']==2 and parent.SETTINGS['mirror']==1 and parent.SETTINGS['prefill']==512 and parent.SETTINGS['esimd']==0,'Actual optimized pair/noESIMD required');m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);source=closure();desc,r=phase_recipe(PHASES[0],m,cfg,inputs,source)
 require(r['native_env']['STRATA_LAYER0_Q8_DIAG']=='0'and r['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']=='1'and r['native_argv'][r['native_argv'].index('--max-context')+1]=='4096','QSA35 exact numeric/startup scope required');require(parent.fresh.PLAN==HERE/'native62-qsa35-session-math-engine-proposal-v1.json','No old SDK substitute');name,cmd=native_command(Path(output)/PHASES[0],desc,r,NS['_CASE'][1],source);return dict(pure_command_construction_passed=True,GPU_launch_performed=False,commands=[dict(phase=PHASES[0],container_name=name,native_command=cmd,recipe=r)],newSDK_actual_ABI_admission_still_required=True)
BASE_RUN=types.FunctionType(natural.NS['run_phase'].__code__,NS,natural.NS['run_phase'].__name__,natural.NS['run_phase'].__defaults__,natural.NS['run_phase'].__closure__)
def run_phase(*args,**kwargs):
 result=BASE_RUN(*args,**kwargs);from qsa35_numeric_raw_v1 import collect
 directory=args[0]/args[1];raw=collect(directory,args[5],SimpleNamespace(**{**globals(),**NS,'parent':parent,'requests':requests}));c.write(directory/'qsa35-numeric-raw.json',raw)
 from source41_pti_raw_budget_v2 import publish_final
 require(publish_final(directory/'raw-budget.json',c.read(directory/'raw-budget.json'),args[0])['passed'],'Final numeric raw budget required');return result
NS.update(__file__=__file__,PLAN=PLAN,PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight,run_phase=run_phase)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})
from source59_trace_pair_startup_v1 import memory_admission
NS['life']=SimpleNamespace(**{**vars(NS['life']),'prepared_ctrl':SimpleNamespace(**{**vars(NS['life'].prepared_ctrl),'memory_admission':memory_admission}),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
P.update(closure=closure,preflight=preflight,NS=NS,PHASES=PHASES,set_case=NS['set_case'],set_runtime=NS['set_runtime']);NS['set_sdk']=parent.set_sdk
for name in('contract','set_case','set_runtime','admission','prerequisite','life','warm_control','mapped','native_pid','stat_snapshot','campaign_stats'):globals()[name]=NS[name]
def main():
 global MODE,PRIOR
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--diagnostic-identity',action='store_true');p.add_argument('--prior-identity-manifest');p.add_argument('--prior-identity-manifest-sha256');o,rest=p.parse_known_args();MODE='diagnostic'if o.diagnostic_identity else'qualifying';require(bool(o.prior_identity_manifest)==bool(o.prior_identity_manifest_sha256)==o.diagnostic_identity,'Diagnostic opt-in requires exact prior pins; fresh default forbids them')
 if o.diagnostic_identity:require(c.sha(o.prior_identity_manifest)==o.prior_identity_manifest_sha256,'Prior manifest changed');PRIOR=c.read(o.prior_identity_manifest)
 identity.configure(MODE,PRIOR);require('--generation'in rest and rest[rest.index('--generation')+1]=='62','Only actual62 successor generation')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 NS['_PORTS']=identity.ports(NS,parent.base.boundary_ports);sys.argv=[sys.argv[0],*rest];parent.main()
if __name__=='__main__':main()
