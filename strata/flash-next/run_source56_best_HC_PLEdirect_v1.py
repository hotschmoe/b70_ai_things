"""One actual56 best compiled HC2/1/1 plus PLEdirect natural control."""
import os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_source56_hc_prefill_tile4_natural_v1 as parent
from source56_best_HC_PLEdirect_contract_v1 import requests,equivalence,baseline61_binding,HC,OFF
c=parent.c;require=parent.require;ROOT=parent.ROOT;HERE=parent.HERE
PLAN=HERE/'source56-best-HC-PLEdirect-recipe-v1.json'
PHASES=('source56-best-HC2-1-1-PLEdirect',)

def closure():
 parent.closure()
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Actual56 best control source changed '+name)
 return c.sha(PLAN)

def phase_recipe(phase,*args):
 require(phase in PHASES,'One actual56 best control phase required');m,r=parent.phase_recipe(phase,*args);argv=r['native_argv'];env=r['native_env']
 for flag,value in (('--ple-io','direct'),('--ple-row-cache','65536'),('--ple-inflight','256')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 env.update({k:'0' for k in OFF});env.update(HC);require(env['STRATA_VERIFY_MIRROR_STAGE_T1']=='1','Same mirror1 source baseline required')
 for row in r['requests']:row['process_temperature']='first_request' if row['ordinal']==1 else 'loaded';row['input_temperature']='first_input' if row['repeat']==0 else 'repeated_input'
 r.update(PLE_mode='direct',producer_binding_sha256=args[-1],source56_vs61_difference_explicit=True,same_source_three_flag_AB_qualified=False,actual_compiled_HC_flags=HC,maximum_campaign_output_tokens=3072,task_completion_qualified=False,cold_cache_qualified=False,physical_IO_bytes_qualified=False,source_lineage='actual55->56; source61 derives52 and omits these HC ports')
 return m,r

def preflight(plan,output):
 require(parent.SETTINGS['generation']==56 and parent.SETTINGS['prefill']==512 and parent.SETTINGS['cards']==2 and parent.SETTINGS['mirror']==1 and parent.SETTINGS['esimd']==0,'Exact existing actual56 paired baseline required')
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);descriptor,r=phase_recipe(PHASES[0],m,cfg,inputs,closure());parent.fresh.binding();baseline61_binding(r);name,cmd=parent.native_command(Path(output)/PHASES[0],descriptor,r,NS['_CASE'][1],closure())
 for flag,path in (('STRATA_HC_Q8_TOKEN_TILE','sycl/src/kernels/hc_native_projection.cpp'),('STRATA_HC_PREFILL_COALESCE','sycl/include/strata/core/native_hc_dispatch.hpp'),('STRATA_HC_PREFILL_TILE4','sycl/src/kernels/hc_native_composition.cpp')):require(flag in (Path(descriptor['source'])/path).read_text(),'Actual56 source must consume '+flag)
 return dict(pure_command_construction_passed=True,GPU_launch_performed=False,commands=[dict(phase=PHASES[0],container_name=name,native_command=cmd,recipe=r)],scope='Actual compiled56 best HC2/1/1+PLEdirect; common61 direct config/history reference, source difference explicit')

NS=dict(parent.NS)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=PLAN,PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})
def main():
 rest=sys.argv[1:];require('--generation' in rest and rest[rest.index('--generation')+1]=='56','Actual56 control only')
 if '--leased' not in rest and '--construction-only' not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*rest,'--leased'])
 old_ns,old_phases=parent.NS,parent.PHASES;old_case,old_rt,old_pre=parent.set_case,parent.set_runtime,parent.preflight
 try:
  parent.NS=NS;parent.PHASES=PHASES;parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight;NS['_PORTS']=parent.base.boundary_ports(NS);parent.main()
 finally:parent.NS=old_ns;parent.PHASES=old_phases;parent.set_case=old_case;parent.set_runtime=old_rt;parent.preflight=old_pre
if __name__=='__main__':main()
