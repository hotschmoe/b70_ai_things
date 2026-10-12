"""Current52 bounded P2048 copy0/1 recorder; existing owned natural boundary."""
import argparse,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native_v2 as parent
import pti_warm_gen4_contract_v1 as two
import pti_warm_owned_exec_v3 as temporal_exec
import collect_source41_lpdisabled_host_v1 as host_reader
require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
GEN=8

def closure():
 parent.closure();m=c.read(HERE/'source52-p2048-profile-recipe-v1.json')
 for name,want in m['files'].items():require(c.sha(ROOT/name)==want,'Recorder source changed '+name)
 return c.sha(HERE/'source52-p2048-profile-recipe-v1.json')

def phase_recipe(phase,*args):
 descriptor,r=parent.phase_recipe(phase,*args)
 runtime=NS['_RUNTIME'];require(runtime is not None and runtime['paused_regular_list_retention_required'],'Actual paused regular-list runtime required')
 require(runtime['clock_alignment_qualified']is False,'Existing intake cannot silently qualify model clock alignment')
 original=r['requests'][0]
 r['requests']=[{**original,'ordinal':1,'max_new':GEN,'window_role':'warmup'},{**original,'ordinal':2,'max_new':GEN,'window_role':'selected'}]
 pid,ticks=NS['_CASE'][1:];session=NS['temporal'].session_name(pid,ticks,closure())
 r.update(PTI_enabled=True,both_arms_H36_enabled=True,host_trace_enabled=True,tool_runtime=runtime,temporal_session=session,selected_request_count=2,maximum_total_output_tokens=2*GEN,capture_kind='current52 P2048 paused warmup plus selected GEN8/16; copy0/1',clock_alignment_qualified=False,full_model_operation_coverage_qualified=False,total_raw_cap_bytes=1<<30,metadata_cap_bytes=512<<20)
 r['native_env'].update(NS['tool'].CLOCK_ENV);r['native_env'].update(NS['tool'].NATIVE_ENV)
 r['native_env'].update(UNITRACE_StartPaused='1',UNITRACE_Session=session,STRATA_CRITICAL_PATH_TRACE='1',STRATA_DECODE_TIMING='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256=closure())
 r['native_argv']=['/pti/bin/unitrace','--start-paused','--session',session,*NS['tool'].OPTIONS,*r['native_argv']]
 require(r['native_env'].get('SYCL_UR_TRACE','0')in('0',''),'No UR trace flood allowed')
 return descriptor,r

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in parent.PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure())
  name,command=parent.native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure())
  require(NS['_RUNTIME']['install_root']+':/pti:ro'in command and '/pti/bin/unitrace'in command[-1] and 'STRATA_DIAGNOSTIC_PTI_FINALIZE=1'in command,'Actual selected tool/finalizer command association required')
  commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'scope':'bounded recorder, no physical alignment/coverage claim'}

def run_phase(*args,**kwargs):
 value=BASE_RUN(*args,**kwargs);out,phase=args[:2];directory=out/phase;recipe=c.read(directory/'recipe.json')
 rows=[r['raw']for r in c.read(directory/'requests.json')]
 pid=NS['native_pid'](value['terminal'],c.read(directory/'command.json'),True)
 files=sorted((directory/'host-traces').glob('host-trace36-*.jsonl'),key=lambda p:int(p.stem.split('-')[-1]))
 proof=host_reader.collect(files,directory/'engine.combined.log',args[5],rows,recipe['stages'],pid);c.write(directory/'current52-host-coverage.json',proof)
 require(proof['passed'],'Actual current52 H36 coverage failed')
 return value

NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType)and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
BASE_RUN=NS['run_phase']
NS.update(__file__=__file__,PLAN=HERE/'source52-p2048-profile-recipe-v1.json',closure=closure,phase_recipe=phase_recipe,preflight=preflight,run_phase=run_phase,temporal_exec=temporal_exec)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':two.requests,'equivalence':two.equivalence})

def main():
 global GEN
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--profile-gen',type=int,choices=(8,16),default=8);opt,rest=p.parse_known_args();GEN=opt.profile_gen
 require('--generation'in rest and rest[rest.index('--generation')+1]=='52' and '--experiment'in rest and rest[rest.index('--experiment')+1]=='copy','Actual52 copy0/1 recorder only')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 old_namespace=parent.NS;old_setters=(parent.set_case,parent.set_runtime,parent.preflight);parent.NS=NS
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.NS=old_namespace;parent.set_case,parent.set_runtime,parent.preflight=old_setters
if __name__=='__main__':main()
