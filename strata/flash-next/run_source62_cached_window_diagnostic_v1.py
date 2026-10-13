"""Current62 bounded cached512-PP/GEN8 capture-free companion with bounded RAW pipe points; original owner."""
import argparse,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native62_v1 as parent
import diagnostic_model_identity_lane_v1 as identity_lane
IDENTITY_MODE="qualifying"
PRIOR=None
import pti_warm_owned_exec_v3 as temporal_exec
require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
GEN=8
COPY=0
PHASES=('source62-cache1536-diagnostic-pilot',)

def closure():
 parent.closure();m=c.read(HERE/'source62-cached-window-diagnostic-recipe-v1.json')
 for name,want in m['files'].items():require(c.sha(ROOT/name)==want,'Recorder source changed '+name)
 return c.sha(HERE/'source62-cached-window-diagnostic-recipe-v1.json')

def phase_recipe(phase,*args):
 descriptor,r=parent.phase_recipe(phase,*args);original=r['requests'][0]
 r['requests']=[{**original,'ordinal':1,'rid':1,'fresh':1,'pin':1536,'max_new':8,'window_role':'warmup','expected_reused':0},{**original,'ordinal':2,'rid':2,'fresh':0,'pin':1536,'max_new':8,'window_role':'selected','expected_reused':1536}]
 argv=r['native_argv']
 for flag,value in (('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 argv[argv.index('--ple-io')+1]='direct';argv[argv.index('--ple-row-cache')+1]='65536'
 r['native_env'].update(STRATA_COPY_ENGINE='0',STRATA_HC_Q8_TOKEN_TILE='2',STRATA_HC_PREFILL_COALESCE='1',STRATA_HC_PREFILL_TILE4='1',STRATA_HC_EXPLICIT_SQRT_DIVIDE='0',STRATA_PREFILL_HOST60='0',STRATA_BATCH_PREFILL_LAYER_COOPERATE='0',STRATA_BATCH_PREFILL_SINGLE_WINDOW='0',STRATA_TRACE='0',STRATA_DECODE_TIMING='0')
 require(r['PTI_enabled']is False and not any(k.startswith('UNITRACE')for k in r['native_env']),'No profiler tool/native trace environment allowed')
 root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010');public=root/'source62-cached-window-profile-v1-public-v1.json';require(c.sha(public)=='f3390633f5205ecfe45b523153b894e7b1a36260d7b5dd67267b55f3235370a3','Exact admitted profile reference required');proof=c.read(public);require(proof['passed']is True and proof['bounded62_cached_window_profile_passed']is True,'Profile reference must be independently admitted')
 prior_root=Path(proof['root']);report=c.read(prior_root/'report.json');require(c.sha(prior_root/'report.json')==proof['producer_report_sha256'],'Actual profile producer changed');key='source62-cache1536-copy0-host-device-profile/recipe.json';require(c.sha(prior_root/key)==report['artifacts'][key],'Actual profile recipe changed');prior=c.read(prior_root/key)
 require(argv==prior['native_argv'][prior['native_argv'].index('/build/strata'):] and r['requests']==prior['requests'],'Exact same cache work/inputs/native CLI required')
 excluded=set(NS['tool'].CLOCK_ENV)|set(NS['tool'].NATIVE_ENV)|{'STRATA_DECODE_TIMING','STRATA_CRITICAL_PATH_TRACE_DIR','STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256'}
 require({k:v for k,v in r['native_env'].items()if k not in excluded}=={k:v for k,v in prior['native_env'].items()if k not in excluded and not k.startswith('UNITRACE')},'Only profiler and aggregate decode-timing instrumentation may differ')
 r.update(diagnostic_identity=(IDENTITY_MODE=='diagnostic'),identity_mode=IDENTITY_MODE,prior_identity=PRIOR,PTI_enabled=False,both_arms_H36_enabled=False,host_trace_enabled=False,selected_request_count=2,maximum_total_output_tokens=16,total_raw_cap_bytes=1<<30,metadata_cap_bytes=512<<20,single_arm_copy_engine=0,capture_kind='Capture-free companion, same optimized source62 cache1536/new512 GEN8; bounded CPU RAW pipe observer retained',profile_public_sha256=c.sha(public),profile_public_path=str(public),profile_recipe_sha256=c.sha(prior_root/key),clock_alignment_qualified=False,full_model_operation_coverage_qualified=False,clean_serving_speed_qualified=False)
 return descriptor,r

def native_command(directory,m,recipe,pid,binding):
 base=parent.BASE_COMMAND
 rebound=types.FunctionType(base.__code__,NS,base.__name__,base.__defaults__,base.__closure__)
 scope={**vars(parent),'BASE_COMMAND':rebound}
 command=types.FunctionType(parent.native_command.__code__,scope,parent.native_command.__name__,parent.native_command.__defaults__,parent.native_command.__closure__)
 return command(directory,m,recipe,pid,binding)

def preflight(plan,output):
 if IDENTITY_MODE=='diagnostic':
  lock=c.read(HERE/'model-lock.json');paths=[ROOT/lock['destination']/x['path']for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')];identity_lane.authenticate(c,PRIOR,HERE/'model-lock.json',lock,paths)
 require(parent.SETTINGS['generation']==62 and parent.SETTINGS['cards']==2 and parent.SETTINGS['prefill']==512 and parent.SETTINGS['mirror']==1,'Actual62 optimized paired existing serial-cache scope required')
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in parent.PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure())
  require(recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and recipe['both_arms_H36_enabled']is False and recipe['host_trace_enabled']is False,'Paired H36 guard retained; H36 OFF required')
  import native61_startup_option_preflight_v1 as startup
  recipe['CPU_startup_option_preflight']=startup.check(recipe,NS['fresh'].SDK/'source')
  name,command=native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure())
  require(not any('/pti' in x or 'UNITRACE' in x or 'STRATA_DIAGNOSTIC_PTI_FINALIZE=1' in x for x in command),'Capture-free native command must omit tracer/finalizer')
  commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'scope':'bounded recorder, no physical alignment/coverage claim'}


NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
BASE_RUN=NS['run_phase']
NS.update(__file__=__file__,PLAN=HERE/'source62-cached-window-diagnostic-recipe-v1.json',closure=closure,phase_recipe=phase_recipe,preflight=preflight,run_phase=BASE_RUN,temporal_exec=temporal_exec,native_command=native_command)
NS['life']=SimpleNamespace(**{**vars(NS['life']),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
from source62_cached_window_profile_contract_v1 import requests as profile_requests
from source62_profile_protocol_raw_v1 import protocol_class
NS['Protocol']=protocol_class(NS['Protocol'],c,NS['life'],closure)
def equivalence(left,right=None):
 require(right is None and len(left)==2 and left[0]['output_ids']==left[1]['output_ids'],'One complete fresh/cached profile owner required')
 return dict(bounded_cached_window_profile_histories_equal=True,actual_reused=left[1]['actual_reused'],actual_new=left[1]['new_prompt_rows'],host_device_clock_qualification_pending=False,profiler_recording_enabled=False,diagnostic_identity=(IDENTITY_MODE=="diagnostic"),speed_qualified=False,stability_qualified=False,full_model_operation_coverage_qualified=False,API_qualified=False,shelf_qualified=False,scope='Current62 fresh pin1536 warmup then cached remaining512 PP+GEN8; metadata capture only')
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':profile_requests,'equivalence':equivalence})

def main():
 global GEN,COPY,PHASES,IDENTITY_MODE,PRIOR
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--diagnostic-identity',action='store_true');p.add_argument('--prior-identity-manifest');p.add_argument('--prior-identity-manifest-sha256');p.add_argument('--profile-gen',type=int,choices=(8,),default=8);p.add_argument('--copy-engine',type=int,choices=(0,),required=True);opt,rest=p.parse_known_args();IDENTITY_MODE='diagnostic'if opt.diagnostic_identity else'qualifying';require(bool(opt.prior_identity_manifest)==bool(opt.prior_identity_manifest_sha256)==opt.diagnostic_identity,'Explicit diagnostic prior pins required; fresh mode forbids them');PRIOR=None
 if opt.diagnostic_identity:
  require(c.sha(opt.prior_identity_manifest)==opt.prior_identity_manifest_sha256,'Exact diagnostic prior manifest changed');PRIOR=c.read(opt.prior_identity_manifest)
 identity_lane.configure(IDENTITY_MODE,PRIOR);GEN=opt.profile_gen;COPY=opt.copy_engine;PHASES=('source62-cache1536-copy%d-diagnostic-pilot'%COPY,)
 require('--generation'in rest and rest[rest.index('--generation')+1]=='62' and '--experiment'in rest and rest[rest.index('--experiment')+1]=='copy','Actual62 copy0/1 recorder only')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 old_phases=parent.PHASES;parent.PHASES=PHASES;NS['PHASES']=PHASES
 old_namespace=parent.NS;old_setters=(parent.set_case,parent.set_runtime,parent.preflight);parent.NS=NS
 try:
  NS['_PORTS']=identity_lane.ports(NS,parent.base.boundary_ports)
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.PHASES=old_phases;parent.NS=old_namespace;parent.set_case,parent.set_runtime,parent.preflight=old_setters
if __name__=='__main__':main()
