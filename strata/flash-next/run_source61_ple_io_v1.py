"""Matched actual61 PLE mmap/direct using existing protected loaded native2 boundary."""
import argparse,copy,hashlib,json,math,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native61_v1 as parent
import source61_ple_identity_ports_v1 as identity
ORIGINAL_PORTS=parent.base.boundary_ports
def boundary_ports(namespace):return identity.ports(namespace,ORIGINAL_PORTS)
from source61_ple_io_contract_v1 import requests, equivalence, command_for, fields
from native61_startup_option_preflight_v1 import check as startup_check

require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
def closure():
 parent.closure()
 manifest=c.read(HERE/'source61-ple-io-recipe-v1.json')
 for name,want in manifest['files'].items():require(c.sha(ROOT/name)==want,'Normal-scheduler actor source changed '+name)
 return c.sha(HERE/'source61-ple-io-recipe-v1.json')

PHASES = ('source61-ple-mmap','source61-ple-direct')

def phase_recipe(phase, *args):
 require(phase in PHASES,'Exact serial/default-native2 comparison phases required')
 m, r = parent.phase_recipe(phase, *args)
 argv = r['native_argv']
 mode='mmap' if phase==PHASES[0] else 'direct'
 for flag,value in (('--ple-io',mode),('--ple-row-cache','65536'),('--ple-inflight','256')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 require('--ple-sync-submit' not in argv,'Default PLE IO worker required')
 r.update(PLE_mode=mode,PLE_raw_row_cache_rows=65536,PLE_IQ4_NL_row_bytes=90,PLE_direct_cache_payload_bytes=65536*90,PLE_cold_residency_qualified=False,PLE_bytes_counters_scope='native io_report rounded physical MB; mmap no native IO report',PLE_cache_payload_excludes_metadata=True)
 for flag, value in (('--batch','2'),('--batch-groups','1'),('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0'),('--adapt-every','0'),('--adapt-swaps','0'),('--adapt-async','0'),('--lookup-chain','0'),('--suffix-draft','0')):
  if flag in argv: argv[argv.index(flag)+1] = value
  else: argv.extend([flag,value])
 env = r['native_env']
 env.update(STRATA_TRACE='1',STRATA_PREFILL_HELP='0',STRATA_PREFILL_CPU_SHARE='0')
 env.update(STRATA_COPY_ENGINE='0',STRATA_VERIFY_MIRROR_STAGE_T1='0',STRATA_FIDELITY_DIAG='0',STRATA_PREFIX_DIAG='0',STRATA_PREFIX_LIFECYCLE_DIAG='0',STRATA_BATCH_FULL_STATE_CHAIN='1',STRATA_BATCH_CHAIN_MIB='8192',STRATA_BATCH_TRANSFER_MIB='512',STRATA_BATCH_PUBLIC_PREFIX='1',STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/NO_CAPTURE_ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_FULL_CACHE_OBSERVER38='0')
 for key in ('STRATA_BATCH_DECODE_SHARE','STRATA_FULL_CACHE_CAPTURE_RIDS38','SYCL_UR_TRACE'):env.pop(key,None)
 corpus=c.read(parent.SETTINGS['corpus']);serial=False
 env['STRATA_BATCH_PREFILL_LAYER_COOPERATE']='1'
 env['STRATA_BATCH_PREFILL_SINGLE_WINDOW']='1'
 env['STRATA_PREFILL_HOST60']='1'
 r['requests']=[dict(ordinal=2*cohort+i+1,rid=2*cohort+i+1,ids=corpus['cases'][i]['ids'],fresh=1,pin=1536,max_new=512 if i==0 else 64,role='long' if i==0 else 'incoming',slot=i,cohort=cohort,expected_reused=0) for cohort in (0,1) for i in range(2)]
 r.update(cooperative_diagnostic_trace=True,budget_runtime_ledger_available=False,clean_performance_qualified=False,coadmission_qualification_open=True,selected_request_count=4,capture_kind='default scheduler native2 incoming-P2048 during ongoing long decode; compact unarmed window events and host token receipts only',maximum_total_output_tokens=1152,natural_EOS=True,observer_enabled=False,raw_observer_armed=False,arrival_after_long_emitted_tokens=256,normal_scheduler_default_decode_share=0.5,native_latency_fairness_scope='one bounded matched pair of histories; no stability promotion',API_qualified=False,full_cache_qualified=False,clean_latency_qualified=False)

 return m,r

def preflight(plan,output):
 if identity.OPTIONS['mode']=='diagnostic':
  lock=c.read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
  identity.prior_binding(c,HERE/'model-lock.json',lock,shards,identity.OPTIONS['prior'],identity.OPTIONS['prior_sha'])
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure());startup_check(recipe,Path(descriptor['source']));argv=recipe['native_argv'];env=recipe['native_env']
  require(argv[argv.index('--batch')+1]=='2' and '--no-prefill-borrow' in argv and '--mtp' not in argv and argv[argv.index('--adapt-swaps')+1]=='0' and argv[argv.index('--adapt-async')+1]=='0','Current native2 public static no-MTP scope')
  require('STRATA_BATCH_DECODE_SHARE' not in env and env['STRATA_VERIFY_MIRROR_STAGE_T1']=='0' and env['STRATA_COPY_ENGINE']=='0' and env['STRATA_FULL_CACHE_OBSERVER38']=='0' and env['STRATA_BATCH_FIDELITY_ARM']=='/results/NO_CAPTURE_ARM','Default scheduler with absent ARM/no heavy capture')
  require(recipe['native_env']['STRATA_PREFILL_HOST60']=='1' and recipe['native_env']['STRATA_BATCH_PREFILL_SINGLE_WINDOW']==recipe['native_env']['STRATA_BATCH_PREFILL_LAYER_COOPERATE']=='1','Fixed diagnostic/pacing flags required')
  require([row['max_new'] for row in recipe['requests']]==[512,64,512,64] and all(len(row['ids'])==2048 for row in recipe['requests']),'Exact authentic P2048 long512/incoming64 requests')
  name,command=parent.native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure());commands.append(dict(phase=phase,container_name=name,native_command=command,recipe=recipe))
 require(len(commands)==2,'Two actual61 PLE mode owners required')
 from source61_ple_io_contract_v1 import matched_recipes
 matched_recipes([x['recipe'] for x in commands])
 return dict(pure_command_construction_passed=True,commands=commands,GPU_launch_performed=False,identity_mode=identity.OPTIONS['mode'],scope='Two actual61 owned PLE mmap/direct processes, identical A/C twice; eight stage observations each; no cold-cache/calibrated GPU/fullmath/cleanperf claim')

from native60_loaded_fairness_protocol_v1 import Protocol

NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=HERE/'source61-ple-io-recipe-v1.json',PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--identity-mode',choices=('qualifying','diagnostic'),default='qualifying');parser.add_argument('--prior-whole-identity');parser.add_argument('--prior-whole-identity-sha256');parser.add_argument('--split-GPU-CPU-close',action='store_true');parser.add_argument('--gpu-child',action='store_true');options,rest=parser.parse_known_args()
 identity.configure(options.identity_mode,options.prior_whole_identity,options.prior_whole_identity_sha256,options.gpu_child)
 if options.split_GPU_CPU_close and '--construction-only' not in rest and not options.gpu_child:
  require(options.identity_mode=='qualifying','GPU/CPU split initially qualifying scope only')
  require('--leased' not in rest and not identity.inherited_GPU_lease(),'Split CPU coordinator must start outside gpu-run')
  import subprocess
  child_args=[x for x in sys.argv[1:] if x!='--split-GPU-CPU-close']+['--gpu-child']
  result=subprocess.run([str(ROOT/'bin/gpu-run'),sys.executable,__file__,*child_args,'--leased'])
  require('--output' in rest,'Explicit output required');output=rest[rest.index('--output')+1]
  # Reconfigure the same exact SDK/source/API namespace without model launch.
  saved=c.read(Path(output)/PHASES[0]/'recipe.json');report=c.read(Path(output)/'report.json')
  parent.configure(saved['natural_settings']);parent.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);NS['fresh']=parent.fresh
  NS['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);NS['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);rt=saved['tool_runtime'];NS['set_runtime'](rt['intake_path'],rt['exact_intake_sha256'])
  identity.close_after_child(sys.modules[__name__],output,result.returncode);return
 require('--generation'in rest and rest[rest.index('--generation')+1]=='61','Only actual61 allowed')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 sys.argv=[sys.argv[0],*rest]
 # Original natural configure/parser and real lifecycle remain unchanged.
 parser=argparse.ArgumentParser(add_help=False)
 for name in('corpus','corpus-sha256'):parser.add_argument('--'+name,required=True)
 _,unused=parser.parse_known_args(rest)
 old=parent.NS; old_phases=parent.PHASES;old_base=parent.base;parent.base=SimpleNamespace(**{**vars(old_base),'boundary_ports':boundary_ports}); parent.NS=NS; parent.PHASES=PHASES
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  old_set_case=parent.set_case;old_preflight=parent.preflight;old_set_runtime=parent.set_runtime
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  try:sys.argv=[sys.argv[0],*rest];parent.main()
  finally:parent.set_case=old_set_case;parent.preflight=old_preflight;parent.set_runtime=old_set_runtime
 finally:parent.NS=old; parent.PHASES=old_phases;parent.base=old_base
if __name__=='__main__':main()
