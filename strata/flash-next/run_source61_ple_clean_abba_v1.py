"""Clean natural source61 PLE mmap/direct ABBA, original owned boundaries."""
import argparse,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native61_v1 as parent
import source61_ple_identity_ports_v1 as identity
from source61_ple_clean_abba_contract_v1 import requests,equivalence,matched_recipes,OFF_ENV
from native61_startup_option_preflight_v1 import check as startup_check
c=parent.c;require=parent.require;ROOT=parent.ROOT;HERE=parent.HERE
PLAN=HERE/'source61-ple-clean-abba-recipe-v1.json'
PHASES=('source61-ple-clean-A1-mmap','source61-ple-clean-B1-direct','source61-ple-clean-B2-direct','source61-ple-clean-A2-mmap')
ORIGINAL_PORTS=parent.base.boundary_ports

def closure():
 parent.closure()
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Clean ABBA source changed '+name)
 return c.sha(PLAN)

def phase_recipe(phase,*args):
 require(phase in PHASES,'Exact clean ABBA phase required');m,r=parent.phase_recipe(phase,*args);argv=r['native_argv'];env=r['native_env'];mode=('mmap','direct','direct','mmap')[PHASES.index(phase)]
 for flag,value in (('--ple-io',mode),('--ple-row-cache','65536'),('--ple-inflight','256'),('--batch','0'),('--prompt-cache','0')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 require('--ple-sync-submit' not in argv,'Default PLE worker required')
 env.update({k:'0' for k in OFF_ENV});env['STRATA_VERIFY_MIRROR_STAGE_T1']='1'
 for key in ('SYCL_UR_TRACE','STRATA_BATCH_DECODE_SHARE'):env.pop(key,None)
 for row in r['requests']:row['process_temperature']='first_request' if row['ordinal']==1 else 'loaded';row['input_temperature']='first_input' if row['repeat']==0 else 'repeated_input'
 r.update(PLE_mode=mode,matched_experiment='PLE_mmap_direct_ABBA',PLE_raw_row_cache_rows=65536,PLE_direct_cache_payload_bytes=5898240,PLE_cache_payload_excludes_metadata=True,PLE_cold_cache_qualified=False,physical_IO_bytes_qualified=False,clean_native_instrumentation_OFF=True,cooperative_diagnostic_trace=False,budget_runtime_ledger_available=False,task_completion_qualified=False,clean_measurement_requires_no_overlapping_build_hash_or_GPU_work=True,maximum_campaign_output_tokens=4*3072)
 return m,r

def preflight(plan,output):
 require(parent.SETTINGS['prefill']==512 and parent.SETTINGS['cards']==2 and parent.SETTINGS['mirror']==1 and parent.SETTINGS['esimd']==0 and parent.SETTINGS['experiment']=='copy' and not parent.SETTINGS.get('profile_a') and not parent.SETTINGS.get('profile_b'),'Exact paired natural baseline512/mirror1/original profile required')
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in PHASES:
  descriptor,r=phase_recipe(phase,m,cfg,inputs,closure());startup_check(r,Path(descriptor['source']));name,cmd=parent.native_command(Path(output)/phase,descriptor,r,NS['_CASE'][1],closure());commands.append(dict(phase=phase,container_name=name,native_command=cmd,recipe=r))
 matched_recipes([x['recipe'] for x in commands])
 from source61_ple_clean_abba_contract_v1 import reference_binding
 reference_binding(commands[0]['recipe']['requests'])
 return dict(pure_command_construction_passed=True,GPU_launch_performed=False,commands=commands,identity_mode='qualifying',ABBA_order=['mmap','direct','direct','mmap'],no_overlapping_IO_requirement='Root must serialize model timing against CPU scans/builds/other jobs; recipe is not an observed clean-environment proof')

def boundary_ports(namespace):return identity.ports(namespace,ORIGINAL_PORTS)
NS=dict(parent.NS)
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=PLAN,PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--split-GPU-CPU-close',action='store_true');parser.add_argument('--gpu-child',action='store_true');options,rest=parser.parse_known_args();identity.configure('qualifying',gpu_child=options.gpu_child)
 if options.split_GPU_CPU_close and '--construction-only' not in rest and not options.gpu_child:
  require('--leased' not in rest and not identity.inherited_GPU_lease(),'Split coordinator must start outside gpu-run')
  import subprocess
  child=[x for x in sys.argv[1:] if x!='--split-GPU-CPU-close']+['--gpu-child'];result=subprocess.run([str(ROOT/'bin/gpu-run'),sys.executable,__file__,*child,'--leased']);out=Path(rest[rest.index('--output')+1]);report=c.read(out/'report.json');r=c.read(out/PHASES[0]/'recipe.json')
  parent.configure(r['natural_settings']);parent.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);NS['fresh']=parent.fresh;NS['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);NS['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);rt=r['tool_runtime'];NS['set_runtime'](rt['intake_path'],rt['exact_intake_sha256']);identity.close_after_child(sys.modules[__name__],out,result.returncode);return
 require('--generation' in rest and rest[rest.index('--generation')+1]=='61','Only actual61 allowed')
 if '--leased' not in rest and '--construction-only' not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 old_ns,old_phases,old_base=parent.NS,parent.PHASES,parent.base;old_case,old_rt,old_pre=parent.set_case,parent.set_runtime,parent.preflight
 try:
  parent.NS=NS;parent.PHASES=PHASES;parent.base=SimpleNamespace(**{**vars(old_base),'boundary_ports':boundary_ports});parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight;NS['_PORTS']=boundary_ports(NS);sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.NS=old_ns;parent.PHASES=old_phases;parent.base=old_base;parent.set_case=old_case;parent.set_runtime=old_rt;parent.preflight=old_pre
if __name__=='__main__':main()
