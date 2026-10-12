"""Current55 bounded P2048 copy0/1 recorder; existing owned natural boundary."""
import argparse,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_source55_hc_prefill_natural_v1 as parent
import pti_warm_gen4_contract_v1 as two
import pti_warm_owned_exec_v3 as temporal_exec
import collect_source41_lpdisabled_host_v1 as host_reader
require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
GEN=8
COPY=0
PHASES=('source55-hc-coalesced-device-profile',)

def closure():
 parent.closure();m=c.read(HERE/'source55-p2048-profile-recipe-v1.json')
 for name,want in m['files'].items():require(c.sha(ROOT/name)==want,'Recorder source changed '+name)
 return c.sha(HERE/'source55-p2048-profile-recipe-v1.json')

def phase_recipe(phase,*args):
 descriptor,r=parent.phase_recipe(phase,*args)
 runtime=NS['_RUNTIME'];require(runtime is not None and runtime['paused_regular_list_retention_required'],'Actual paused regular-list runtime required')
 require(runtime['clock_alignment_qualified']is False,'Existing intake cannot silently qualify model clock alignment')
 original=r['requests'][0]
 r['requests']=[{**original,'ordinal':1,'max_new':GEN,'window_role':'warmup'},{**original,'ordinal':2,'max_new':GEN,'window_role':'selected'}]
 pid,ticks=NS['_CASE'][1:];session=NS['temporal'].session_name(pid,ticks,closure())
 r.update(PTI_enabled=True,both_arms_H36_enabled=False,host_trace_enabled=False,tool_runtime=runtime,temporal_session=session,selected_request_count=2,maximum_total_output_tokens=2*GEN,capture_kind='current55 P2048 warmup/selected GEN8; HCtile2/coalescing1/copy0',clock_alignment_qualified=False,full_model_operation_coverage_qualified=False,total_raw_cap_bytes=1<<30,metadata_cap_bytes=512<<20)
 r['native_env'].update(NS['tool'].CLOCK_ENV);r['native_env'].update(NS['tool'].NATIVE_ENV)
 r['native_env'].update(UNITRACE_StartPaused='1',UNITRACE_Session=session,STRATA_CRITICAL_PATH_TRACE='0',STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=closure())
 require(COPY==0,'Copy0 fixed for HC coalesced profile');r['native_env'].update(STRATA_COPY_ENGINE='0',STRATA_HC_Q8_TOKEN_TILE='2',STRATA_HC_PREFILL_COALESCE='1')
 r.update(single_arm_copy_engine=COPY,independent_diagnostic_experiments=1,original_individual_root_cap_bytes=1<<30)
 r.update(Chrome_CPU_host_lane_recorded=False,Chrome_device_only=True,H36_coverage_qualified=False,host_lane='not recorded; device-only capture omits Chrome CPU API lane and H36 remains OFF')
 options=[x for x in NS['tool'].OPTIONS if x not in('--chrome-call-logging','--chrome-kernel-logging')]
 require(options.count('--chrome-device-logging')==1 and not any(r['native_env'].get(k)for k in('UNITRACE_ChromeCallLogging','UNITRACE_ChromeKernelLogging')),'Device-only flags and clean inherited profiler environment required')
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
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in parent.PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure())
  require(recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0' and recipe['both_arms_H36_enabled']is False and recipe['host_trace_enabled']is False,'Paired H36 guard retained; H36 OFF required')
  import native52_startup_option_preflight_v1 as startup
  recipe['CPU_startup_option_preflight']=startup.check(recipe,NS['fresh'].SDK/'source')
  name,command=native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure())
  require(NS['_RUNTIME']['install_root']+':/pti:ro'in command and '/pti/bin/unitrace'in command[-1] and 'STRATA_DIAGNOSTIC_PTI_FINALIZE=1'in command,'Actual selected tool/finalizer command association required')
  commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'scope':'bounded recorder, no physical alignment/coverage claim'}


NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
BASE_RUN=NS['run_phase']
NS.update(__file__=__file__,PLAN=HERE/'source55-p2048-profile-recipe-v1.json',closure=closure,phase_recipe=phase_recipe,preflight=preflight,run_phase=BASE_RUN,temporal_exec=temporal_exec,native_command=native_command)
NS['life']=SimpleNamespace(**{**vars(NS['life']),'contract':SimpleNamespace(**{**vars(NS['life'].contract),'native_command':native_command})})
def equivalence(left,right=None):
 require(len(left)==2 and right is None and __import__('serial37_canonical_json_v3').canonical(left[0])==__import__('serial37_canonical_json_v3').canonical(left[1]),'One complete warmup/selected arm required');return {'H36_OFF_ON_equivalence_passed':False,'warmup_selected_outputs_equal':True,'matched_trace_arms_qualified':False,'scope':'one independent current55 P2048 GEN%d copy%d diagnostic; mate/join separate'%(GEN,COPY)}
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':two.requests,'equivalence':equivalence})

def main():
 global GEN,COPY,PHASES
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--profile-gen',type=int,choices=(8,),default=8);p.add_argument('--copy-engine',type=int,choices=(0,),required=True);opt,rest=p.parse_known_args();GEN=opt.profile_gen;COPY=opt.copy_engine;PHASES=('source55-hc-coalesced-device-profile',)
 require('--generation'in rest and rest[rest.index('--generation')+1]=='55' and '--experiment'in rest and rest[rest.index('--experiment')+1]=='HCprefill','Actual55 coalescing1 tile2 copy0 recorder only')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 old_phases=parent.PHASES;parent.PHASES=PHASES;NS['PHASES']=PHASES
 old_namespace=parent.NS;old_setters=(parent.set_case,parent.set_runtime,parent.preflight);parent.NS=NS
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.PHASES=old_phases;parent.NS=old_namespace;parent.set_case,parent.set_runtime,parent.preflight=old_setters
if __name__=='__main__':main()
