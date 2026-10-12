"""Normal-scheduler native2 arrival/token-gap probe; existing owned model boundary."""
import argparse,copy,hashlib,json,math,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native58_v1 as parent
from source58_native2_cooperative_contract_v1 import requests, equivalence, command_for, fields
from native58_startup_option_preflight_v1 import check as startup_check

require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
def closure():
 parent.closure()
 manifest=c.read(HERE/'source58-native2-cooperative-recipe-v1.json')
 for name,want in manifest['files'].items():require(c.sha(ROOT/name)==want,'Normal-scheduler actor source changed '+name)
 return c.sha(HERE/'source58-native2-cooperative-recipe-v1.json')

PHASES = ('source58-native2-cooperative-OFF','source58-native2-cooperative-ON')

def phase_recipe(phase, *args):
 require(phase in PHASES,'Exact serial/default-native2 comparison phases required')
 m, r = parent.phase_recipe(phase, *args)
 argv = r['native_argv']
 for flag, value in (('--batch','2'),('--batch-groups','1'),('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0'),('--adapt-every','0'),('--adapt-swaps','0'),('--adapt-async','0'),('--lookup-chain','0'),('--suffix-draft','0')):
  if flag in argv: argv[argv.index(flag)+1] = value
  else: argv.extend([flag,value])
 env = r['native_env']
 env.update(STRATA_TRACE='1',STRATA_PREFILL_HELP='0',STRATA_PREFILL_CPU_SHARE='0')
 env.update(STRATA_COPY_ENGINE='0',STRATA_VERIFY_MIRROR_STAGE_T1='0',STRATA_FIDELITY_DIAG='0',STRATA_PREFIX_DIAG='0',STRATA_PREFIX_LIFECYCLE_DIAG='0',STRATA_BATCH_FULL_STATE_CHAIN='1',STRATA_BATCH_CHAIN_MIB='8192',STRATA_BATCH_TRANSFER_MIB='512',STRATA_BATCH_PUBLIC_PREFIX='1',STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/NO_CAPTURE_ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',STRATA_FULL_CACHE_OBSERVER38='0')
 for key in ('STRATA_BATCH_DECODE_SHARE','STRATA_FULL_CACHE_CAPTURE_RIDS38','SYCL_UR_TRACE'):env.pop(key,None)
 corpus=c.read(parent.SETTINGS['corpus']);serial=False
 env['STRATA_BATCH_PREFILL_LAYER_COOPERATE']=str(PHASES.index(phase))
 r['requests']=[dict(ordinal=i+1,rid=i+1,ids=corpus['cases'][i]['ids'],fresh=1,pin=1536,max_new=512 if i==0 else 64,role='long' if i==0 else 'incoming',slot=-1 if serial else i,cohort=i if serial else 0,expected_reused=0) for i in range(2)]
 r.update(cooperative_diagnostic_trace=True,budget_runtime_ledger_available=False,clean_performance_qualified=False,coadmission_qualification_open=True,selected_request_count=2,capture_kind='default scheduler native2 incoming-P2048 during ongoing long decode; compact unarmed window events and host token receipts only',maximum_total_output_tokens=576,natural_EOS=True,observer_enabled=False,raw_observer_armed=False,arrival_after_long_emitted_tokens=256,normal_scheduler_default_decode_share=0.5,native_latency_fairness_scope='one bounded matched pair of histories; no stability promotion',API_qualified=False,full_cache_qualified=False,clean_latency_qualified=False)

 return m,r

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure());startup_check(recipe,Path(descriptor['source']));argv=recipe['native_argv'];env=recipe['native_env']
  require(argv[argv.index('--batch')+1]=='2' and '--no-prefill-borrow' in argv and '--mtp' not in argv and argv[argv.index('--adapt-swaps')+1]=='0' and argv[argv.index('--adapt-async')+1]=='0','Current native2 public static no-MTP scope')
  require('STRATA_BATCH_DECODE_SHARE' not in env and env['STRATA_VERIFY_MIRROR_STAGE_T1']=='0' and env['STRATA_COPY_ENGINE']=='0' and env['STRATA_FULL_CACHE_OBSERVER38']=='0' and env['STRATA_BATCH_FIDELITY_ARM']=='/results/NO_CAPTURE_ARM','Default scheduler with absent ARM/no heavy capture')
  require([row['max_new'] for row in recipe['requests']]==[512,64] and all(len(row['ids'])==2048 for row in recipe['requests']),'Exact authentic P2048 long512/incoming64 requests')
  name,command=parent.native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure());commands.append(dict(phase=phase,container_name=name,native_command=command,recipe=recipe))
 require(commands[0]['recipe']['native_argv']==commands[1]['recipe']['native_argv'],'Matched58 current args required')
 left=dict(commands[0]['recipe']['native_env']);right=dict(commands[1]['recipe']['native_env']);require(left.pop('STRATA_BATCH_PREFILL_LAYER_COOPERATE')=='0' and right.pop('STRATA_BATCH_PREFILL_LAYER_COOPERATE')=='1' and left==right,'Sole cooperative scheduling flag difference')
 return dict(pure_command_construction_passed=True,commands=commands,GPU_launch_performed=False,scope='Diagnostic actual58 normal-scheduler latency/histories; observed rows/service; budget/cancellation/coadmission/cleanperf remain unqualified; actual root runtime required')

from native2_fairness_protocol_v1 import Protocol

NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=HERE/'source58-native2-cooperative-recipe-v1.json',PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})

def main():
 rest=sys.argv[1:]
 require('--generation'in rest and rest[rest.index('--generation')+1]=='58','Only actual58 allowed')
 if '--leased'not in rest and '--construction-only'not in rest:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 # Original natural configure/parser and real lifecycle remain unchanged.
 parser=argparse.ArgumentParser(add_help=False)
 for name in('corpus','corpus-sha256'):parser.add_argument('--'+name,required=True)
 _,unused=parser.parse_known_args(rest)
 old=parent.NS; old_phases=parent.PHASES; parent.NS=NS; parent.PHASES=PHASES
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  old_set_case=parent.set_case;old_preflight=parent.preflight;old_set_runtime=parent.set_runtime
  parent.set_case=NS['set_case'];parent.set_runtime=NS['set_runtime'];parent.preflight=preflight
  try:sys.argv=[sys.argv[0],*rest];parent.main()
  finally:parent.set_case=old_set_case;parent.preflight=old_preflight;parent.set_runtime=old_set_runtime
 finally:parent.NS=old; parent.PHASES=old_phases
if __name__=='__main__':main()
