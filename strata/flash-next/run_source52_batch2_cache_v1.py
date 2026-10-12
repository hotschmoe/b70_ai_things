"""Six-request native batch2 cache isolation probe; existing owned model boundary."""
import argparse,copy,hashlib,json,math,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native_v2 as parent
from source52_batch2_cache_contract_v1 import requests, equivalence, command_for, fields
from native52_startup_option_preflight_v1 import check as startup_check

require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
FIXTURE=None;FIXTURE_SHA=None

def fixture():
 require(FIXTURE is not None and c.sha(FIXTURE)==FIXTURE_SHA,'Immutable authentic cache fixture required')
 value=c.read(FIXTURE);a=value['A'];b=value['B'];pin=value['pin']
 require(pin==1536 and len(a)==2048 and 1536<len(b)<=3072 and a[:pin]==b[:pin] and a!=b,'Authentic divergent shared1536 prefix required')
 require(value['actual_GPU_touch']is False and value['actual_model_payload_read']is False,'Tokenizer-only producer required')
 corpus=c.read(parent.SETTINGS['corpus']);require(a==corpus['cases'][0]['ids'],'A must be original P2048 natural case')
 require(value['producer_sha256']==c.sha(HERE/'render_source52_serial_cache_fixture_v2.py') and value['input_corpus_sha256']==c.sha(parent.SETTINGS['corpus']),'Actual rendered fixture producer/corpus differs')
 require(value['software_sha256']==corpus['software_sha256'] and value['tokenizer_sha256']==corpus['tokenizer_sha256'],'Pinned current tokenizer/template association required')
 return value

def closure():
 parent.closure();fixture()
 manifest=c.read(HERE/'source52-batch2-cache-recipe-v1.json')
 for name,want in manifest['files'].items():require(c.sha(ROOT/name)==want,'New cache actor source changed '+name)
 return c.sha(HERE/'source52-batch2-cache-recipe-v1.json')

PHASES = ('source52-native-batch2-cache',)

def phase_recipe(phase, *args):
 require(phase == PHASES[0], 'One actual owned native phase required')
 m, r = parent.phase_recipe(phase, *args)
 f = fixture(); argv = r['native_argv']
 for flag, value in (('--batch','2'),('--batch-groups','1'),('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0'),('--adapt-every','0'),('--lookup-chain','0'),('--suffix-draft','0')):
  if flag in argv: argv[argv.index(flag)+1] = value
  else: argv.extend([flag,value])
 env = r['native_env']
 env.update(STRATA_COPY_ENGINE='0', STRATA_VERIFY_MIRROR_STAGE_T1='0', STRATA_FIDELITY_DIAG='0', STRATA_PREFIX_DIAG='0', STRATA_PREFIX_LIFECYCLE_DIAG='0', STRATA_BATCH_FULL_STATE_CHAIN='1', STRATA_BATCH_CHAIN_MIB='8192', STRATA_BATCH_TRANSFER_MIB='512', STRATA_BATCH_PUBLIC_PREFIX='1', STRATA_BATCH_FIDELITY_DIAG='1', STRATA_BATCH_FIDELITY_ARM='/results/ARM', STRATA_BATCH_FIDELITY_DIR='/results/captures', STRATA_FULL_CACHE_OBSERVER38='1', STRATA_FULL_CACHE_CAPTURE_RIDS38='1,2,3,4,5,6')
 env.pop('SYCL_UR_TRACE', None)
 r['requests'] = []
 for ordinal, (role,slot,fresh,reused,cohort) in enumerate((('A',-1,1,0,0),('B',-1,1,0,1),('A',0,1,0,2),('B',1,1,0,2),('B',0,0,1536,3),('A',1,0,1536,3)),1):
  r['requests'].append(dict(ordinal=ordinal,rid=ordinal,ids=f[role],fresh=fresh,pin=1536,max_new=8,role=role,slot=slot,cohort=cohort,expected_reused=reused))
 r.update(capture_kind='serial A/B controls, cold AB batch, cached BA batch; original bounded SBF all49 observer',maximum_total_output_tokens=48,natural_EOS=True,observer_enabled=True,cache_fixture_path=str(FIXTURE),cache_fixture_sha256=FIXTURE_SHA,bounded_basic_cache_only=True,native_two_stream_qualification_pending=True,API_qualified=False,full_cache_qualified=False,clean_latency_qualified=False)
 return m,r

def preflight(plan, output):
 m = c.read(Path(plan['template_path'])/'prepared.json'); cfg = c.read(Path(plan['template_path'])/'server-config.json'); inputs = c.read(plan['inputs_path'])
 descriptor, recipe = phase_recipe(PHASES[0],m,cfg,inputs,closure())
 startup_check(recipe,Path(descriptor['source']))
 argv=recipe['native_argv']; env=recipe['native_env']
 require(argv[argv.index('--batch')+1]=='2' and argv[argv.index('--batch-groups')+1]=='1' and '--no-prefill-borrow' in argv and '--mtp' not in argv, 'Strict two-slot static no-MTP/no-borrow scope')
 require(env['STRATA_VERIFY_MIRROR_STAGE_T1']=='0' and env['STRATA_PF_GROUPED_F16']=='0' and env['STRATA_PF_ESIMD_F16']=='0', 'Existing exact52 safe math route')
 name, command=parent.native_command(Path(output)/PHASES[0],descriptor,recipe,NS['_CASE'][1],closure())
 return dict(pure_command_construction_passed=True,commands=[dict(phase=PHASES[0],container_name=name,native_command=command,recipe=recipe)],GPU_launch_performed=False,scope='bounded current52 native2 cache probe; no actual qualification yet')

class Protocol(parent.NS['Protocol']):
 @staticmethod
 def is_protocol(line):
  return line.startswith(('READY','INFO ','ERR','FATAL','RESUME ','REUSED ','PP ','T ','LP ','DONE ','BT ','BADM ','BDONE ','SBF ','FC38 '))
 def __init__(self,command,directory,*args,**kwargs):
  from batch_numerical_protocol_v2 import Roster
  self.schedule=c.read(Path(directory)/'recipe.json')['requests']; self.batch=Roster(2); self.completed={}
  super().__init__(command,directory,*args,**kwargs)
  for line in self.stdout:
   if line.startswith('INFO '): self.batch.consume(line)
  self.batch.capacity()
  (self.directory/'ARM').write_text('ARM source52 native2 six original bounded observations\n',encoding='ascii')
 def request(self,label,ids,**kwargs):
  import re,time
  index=int(label.removeprefix('request-'))-1; want=self.schedule[index]
  require(want['ids']==ids, 'Actual original owner input differs')
  if index in self.completed: return self.completed[index]
  group=[(i,row) for i,row in enumerate(self.schedule) if row['cohort']==want['cohort']]
  start=time.monotonic(); deadline=start+1800; lines=[]; times=[]; sends={}
  for i,row in group:
   if row['slot']>=0: self.batch.submit(row['rid'],row['slot'],row['ids'],row['max_new'])
   sends[i]=command_for(row); self.send(sends[i])
  while True:
   stamp,line=self.q.get(timeout=max(.001,deadline-time.monotonic()))
   require(line is not None, 'Native EOF/quota failure within cohort')
   require(time.monotonic()<deadline, 'Bounded native cohort deadline')
   require(not line.startswith(('ERR','FATAL')), 'Native cohort refusal: '+line)
   lines.append(line); times.append(stamp-start)
   if want['slot']>=0: self.batch.consume(line)
   if want['slot']<0:
    if line.startswith('DONE ') and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rid')==str(want['rid']): break
   elif all(self.batch.requests[row['rid']]['done'] for _,row in group): break
  events=[dict(line=line,seconds=t) for line,t in zip(lines,times)]
  c.write(self.directory/('cohort-'+str(want['cohort'])+'.json'),dict(cohort=want['cohort'],commands=[sends[i] for i,_ in group],events=events))
  for i,row in group:
   own=[(line,t) for line,t in zip(lines,times) if dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rid')==str(row['rid']) and line.startswith(('T ','BT ','DONE ','BADM ','BDONE '))]
   emitted=[int(line.split()[1 if line.startswith('T ') else 2]) for line,_ in own if line.startswith(('T ','BT '))]
   terminal=next(line for line,_ in reversed(own) if line.startswith('DONE ' if row['slot']<0 else 'BDONE '))
   raw=dict(command=sends[i],ids=row['ids'],fresh=row['fresh'],pin=row['pin'],rid=row['rid'],slot=row['slot'],cohort=row['cohort'],output_ids=emitted,stdout=[x for x,_ in own],stdout_seconds=[t for _,t in own],done=terminal,LP=[],stop_sent=False,cancel_requested=None,capture_directory=str((self.directory/'captures').resolve()),phase_directory=str(self.directory.resolve()))
   self.completed[i]=raw
   c.write(self.directory/('request-'+str(i+1)+'.json'),raw)
  return self.completed[index]

NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=HERE/'source52-batch2-cache-recipe-v1.json',PHASES=PHASES,closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight)
NS['contract']=SimpleNamespace(**{**vars(NS['contract']),'requests':requests,'equivalence':equivalence})

def main():
 global FIXTURE,FIXTURE_SHA
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--cache-fixture',type=Path,required=True);p.add_argument('--cache-fixture-sha256',required=True);opt,rest=p.parse_known_args();FIXTURE=opt.cache_fixture.resolve();FIXTURE_SHA=opt.cache_fixture_sha256
 require('--generation'in rest and rest[rest.index('--generation')+1]=='52','Only actual52 allowed')
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
