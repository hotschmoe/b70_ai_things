"""Current62 bounded cached512-PP/GEN8 host/device RAW recorder; original owner."""
import argparse,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native62_v1 as parent
import pti_warm_owned_exec_v3 as temporal_exec
require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
GEN=8
COPY=0
PHASES=('source62-cache1536-host-device-profile',)

def closure():
 parent.closure();m=c.read(HERE/'source62-cached-window-profile-recipe-v1.json')
 for name,want in m['files'].items():require(c.sha(ROOT/name)==want,'Recorder source changed '+name)
 return c.sha(HERE/'source62-cached-window-profile-recipe-v1.json')

def phase_recipe(phase,*args):
 descriptor,r=parent.phase_recipe(phase,*args)
 runtime=NS['_RUNTIME'];require(runtime is not None and runtime['paused_regular_list_retention_required'],'Actual paused regular-list runtime required')
 require(runtime['clock_alignment_qualified']is False,'Existing intake cannot silently qualify model clock alignment')
 original=r['requests'][0]
 r['requests']=[{**original,'ordinal':1,'rid':1,'fresh':1,'pin':1536,'max_new':8,'window_role':'warmup','expected_reused':0},{**original,'ordinal':2,'rid':2,'fresh':0,'pin':1536,'max_new':8,'window_role':'selected','expected_reused':1536}]
 argv=r['native_argv']
 for flag,value in (('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 r.update(profile_selected_scope='Cached prefix1536, remaining512 PP plus GEN8, full P2048 input retained; not uncached full-prefill',explicit_request_RIDs=[1,2],internal_sampling_coverage_qualified=False,paired_H36_coverage_qualified=False,full_stage_handle_mapping_qualified=False)
 pid,ticks=NS['_CASE'][1:];session=NS['temporal'].session_name(pid,ticks,closure())
 r.update(PTI_enabled=True,both_arms_H36_enabled=False,host_trace_enabled=False,tool_runtime=runtime,temporal_session=session,selected_request_count=2,maximum_total_output_tokens=2*GEN,capture_kind='current62 P2048 warmup then cached1536/remaining512PP/GEN8, actual API/device RAW callbacks',clock_alignment_qualified=False,full_model_operation_coverage_qualified=False,total_raw_cap_bytes=1<<30,metadata_cap_bytes=512<<20)
 r['native_env'].update(NS['tool'].CLOCK_ENV);r['native_env'].update(NS['tool'].NATIVE_ENV)
 r['native_env'].update(UNITRACE_StartPaused='1',UNITRACE_Session=session,STRATA_CRITICAL_PATH_TRACE='0',STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=closure())
 r['native_env'].update(STRATA_COPY_ENGINE=str(COPY),STRATA_HC_Q8_TOKEN_TILE='2',STRATA_HC_PREFILL_COALESCE='1',STRATA_HC_PREFILL_TILE4='1',STRATA_HC_EXPLICIT_SQRT_DIVIDE='0',STRATA_PREFILL_HOST60='0',STRATA_BATCH_PREFILL_LAYER_COOPERATE='0',STRATA_BATCH_PREFILL_SINGLE_WINDOW='0',STRATA_TRACE='0')
 argv=r['native_argv'];argv[argv.index('--ple-io')+1]='direct';argv[argv.index('--ple-row-cache')+1]='65536'
 r.update(actual_optimized_HC_consumers_enabled=True,explicit_sqrt_divide=0,PLE_mode='direct',mirror_stage_T1_scope='ON; optimized GPU-resident stage1 T1 path; host fallback remains source-governed')
 r.update(single_arm_copy_engine=COPY,independent_diagnostic_experiments=1,original_individual_root_cap_bytes=1<<30)
 r.update(Chrome_CPU_host_lane_recorded=True,Chrome_device_only=False,H36_coverage_qualified=False,host_lane='Existing L0 CPU API callbacks plus RAW native protocol receive points; paired H36/internal preparation/sampling remains OFF/unqualified')
 options=[x for x in NS['tool'].OPTIONS if x!='--chrome-kernel-logging']
 if '--chrome-call-logging' not in options:options.append('--chrome-call-logging')
 require(options.count('--chrome-device-logging')==options.count('--chrome-call-logging')==1,'Actual CPU API and device callbacks required')
 r['native_argv']=['/pti/bin/unitrace','--start-paused','--session',session,*options,*r['native_argv']]
 require(r['native_env'].get('SYCL_UR_TRACE','0')in('0',''),'No UR trace flood allowed')
 return descriptor,r

def native_command(directory,m,recipe,pid,binding):
 base=parent.BASE_COMMAND
 rebound=types.FunctionType(base.__code__,NS,base.__name__,base.__defaults__,base.__closure__)
 scope={**vars(parent),'BASE_COMMAND':rebound}
 command=types.FunctionType(parent.native_command.__code__,scope,parent.native_command.__name__,parent.native_command.__defaults__,parent.native_command.__closure__)
 return command(directory,m,recipe,pid,binding)

def preflight(plan,output):
 require(parent.SETTINGS['generation']==62 and parent.SETTINGS['cards']==2 and parent.SETTINGS['prefill']==512 and parent.SETTINGS['mirror']==1,'Actual62 optimized paired existing serial-cache scope required')
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in parent.PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure())
  require(recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and recipe['both_arms_H36_enabled']is False and recipe['host_trace_enabled']is False,'Paired H36 guard retained; H36 OFF required')
  import native61_startup_option_preflight_v1 as startup
  recipe['CPU_startup_option_preflight']=startup.check(recipe,NS['fresh'].SDK/'source')
  name,command=native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure())
  require(NS['_RUNTIME']['install_root']+':/pti:ro'in command and '/pti/bin/unitrace'in command[-1] and 'STRATA_DIAGNOSTIC_PTI_FINALIZE=1'in command,'Actual selected tool/finalizer command association required')
  commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'scope':'bounded recorder, no physical alignment/coverage claim'}


NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
BASE_RUN=NS['run_phase']
NS.update(__file__=__file__,PLAN=HERE/'source62-cached-window-profile-recipe-v1.json',closure=closure,phase_recipe=phase_recipe,preflight=preflight,run_phase=BASE_RUN,temporal_exec=temporal_exec,native_command=native_command)
NS['life']=SimpleNamespace(**{**vars(NS['life']),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
from source62_cached_window_profile_contract_v1 import requests as profile_requests
from source62_profile_protocol_raw_v1 import protocol_class
NS['Protocol']=protocol_class(NS['Protocol'],c,NS['life'],closure)
def equivalence(left,right=None):
 require(right is None and len(left)==2 and left[0]['output_ids']==left[1]['output_ids'],'One complete fresh/cached profile owner required')
 return dict(bounded_cached_window_profile_histories_equal=True,actual_reused=left[1]['actual_reused'],actual_new=left[1]['new_prompt_rows'],host_device_clock_qualification_pending=True,full_model_operation_coverage_qualified=False,API_qualified=False,shelf_qualified=False,scope='Current62 fresh pin1536 warmup then cached remaining512 PP+GEN8; metadata capture only')
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':profile_requests,'equivalence':equivalence})

def main():
 global GEN,COPY,PHASES
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--profile-gen',type=int,choices=(8,),default=8);p.add_argument('--copy-engine',type=int,choices=(0,1),required=True);opt,rest=p.parse_known_args();GEN=opt.profile_gen;COPY=opt.copy_engine;PHASES=('source62-cache1536-copy%d-host-device-profile'%COPY,)
 require('--generation'in rest and rest[rest.index('--generation')+1]=='62' and '--experiment'in rest and rest[rest.index('--experiment')+1]=='copy','Actual62 copy0/1 recorder only')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 old_phases=parent.PHASES;parent.PHASES=PHASES;NS['PHASES']=PHASES
 old_namespace=parent.NS;old_setters=(parent.set_case,parent.set_runtime,parent.preflight);parent.NS=NS
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.PHASES=old_phases;parent.NS=old_namespace;parent.set_case,parent.set_runtime,parent.preflight=old_setters
if __name__=='__main__':main()
