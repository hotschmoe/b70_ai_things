"""Bounded serial cache latency experiment on actual52; existing natural owner boundary."""
import argparse,copy,hashlib,json,os,sys,types
from pathlib import Path
from types import SimpleNamespace
import run_natural_p2048_native_v2 as parent
import serial_prefix_qualification_v8 as numeric

require=parent.require;c=parent.c;ROOT=parent.ROOT;HERE=parent.HERE
FIXTURE=None;FIXTURE_SHA=None

def fixture():
 require(FIXTURE is not None and c.sha(FIXTURE)==FIXTURE_SHA,'Immutable authentic cache fixture required')
 value=c.read(FIXTURE);a=value['A'];b=value['B'];pin=value['pin']
 require(pin==1536 and len(a)==2048 and 1536<len(b)<=3072 and a[:pin]==b[:pin] and a!=b,'Authentic divergent shared1536 prefix required')
 require(value['actual_GPU_touch']is False and value['actual_model_payload_read']is False,'Tokenizer-only producer required')
 corpus=c.read(parent.SETTINGS['corpus']);require(a==corpus['cases'][0]['ids'],'A must be original P2048 natural case')
 require(value['producer_sha256']==c.sha(HERE/'render_source52_serial_cache_fixture_v1.py') and value['input_corpus_sha256']==c.sha(parent.SETTINGS['corpus']),'Actual rendered fixture producer/corpus differs')
 require(value['software_sha256']==corpus['software_sha256'] and value['tokenizer_sha256']==corpus['tokenizer_sha256'],'Pinned current tokenizer/template association required')
 return value

def closure():
 parent.closure();fixture()
 manifest=c.read(HERE/'source52-serial-cache-latency-recipe-v1.json')
 for name,want in manifest['files'].items():require(c.sha(ROOT/name)==want,'New cache actor source changed '+name)
 return c.sha(HERE/'source52-serial-cache-latency-recipe-v1.json')

def phase_recipe(phase,*args):
 m,r=parent.phase_recipe(phase,*args);f=fixture();cold=phase==parent.PHASES[0]
 argv=r['native_argv']
 for flag,value in(('--prompt-cache','3'),('--prompt-cache-root','1'),('--prompt-cache-every','1000000000'),('--conversation-cache-mib','0'),('--conversation-cache-slots','0')):
  if flag in argv:argv[argv.index(flag)+1]=value
  else:argv.extend([flag,value])
 env=r['native_env'];env.update(STRATA_COPY_ENGINE='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='0',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM')
 r['requests']=[{'ordinal':i+1,'ids':f[key],'fresh':int(cold),'pin':1536,'max_new':64,'role':key,'expected_reused':0 if cold or i==0 else 1536}for i,key in enumerate(('A','A','B','B','A','B'))]
 r.update(capture_kind='serial52 matched cold/reuse shared1536; full first head only',maximum_total_output_tokens=384,natural_EOS=False,observer_enabled=True,cache_fixture_path=str(FIXTURE),cache_fixture_sha256=FIXTURE_SHA,bounded_basic_cache_only=True)
 return m,r

def preflight(plan,output):
 m=c.read(Path(plan['template_path'])/'prepared.json');cfg=c.read(Path(plan['template_path'])/'server-config.json');inputs=c.read(plan['inputs_path']);commands=[]
 for phase in parent.PHASES:
  descriptor,recipe=phase_recipe(phase,m,cfg,inputs,closure())
  require(len(recipe['requests'])==6 and recipe['native_env']['STRATA_COPY_ENGINE']=='0' and recipe['native_env']['STRATA_PF_GROUPED_F16']=='0' and recipe['native_env']['STRATA_PF_ESIMD_F16']=='0','Only declared serial cache profile')
  argv=recipe['native_argv'];require(argv[argv.index('--batch')+1]=='0' and '--no-prefill-borrow'in argv and '--mtp'not in argv,'Serial static no-MTP/no-borrow scope')
  name,command=parent.native_command(Path(output)/phase,descriptor,recipe,NS['_CASE'][1],closure());commands.append({'phase':phase,'container_name':name,'native_command':command,'recipe':recipe})
 require(commands[0]['recipe']['native_argv']==commands[1]['recipe']['native_argv'] and commands[0]['recipe']['native_env']==commands[1]['recipe']['native_env'],'Only request fresh state may differ')
 return {'pure_command_construction_passed':True,'commands':commands,'GPU_launch_performed':False,'bounded_basic_cache_only':True}

class Protocol(parent.NS['Protocol']):
 def __init__(self,command,directory,*args,**kwargs):
  self.roster=c.read(Path(directory)/'recipe.json')['requests'];super().__init__(command,directory,*args,**kwargs)
  (self.directory/'ARM').write_text('ARM serial52 six matched cache requests\n',encoding='ascii')
 def request(self,label,ids,**kwargs):
  index=int(label.removeprefix('request-'))-1;want=self.roster[index]
  require(ids==want['ids'],'Owner original request IDs differ')
  kwargs.update(fresh=want['fresh'],pin=want['pin'],max_new=want['max_new'])
  raw=super().request(label,ids,**kwargs);raw['capture_directory']=str((self.directory/'captures').resolve());c.write(self.directory/(label+'.json'),raw);return raw

def requests(rows,expected):
 require(len(rows)==len(expected)==6,'Six completed bounded cache requests required');views=[]
 for row,want in zip(rows,expected):
  raw=row['raw'];require(row['ordinal']==want['ordinal'] and raw['ids']==want['ids'] and raw['fresh']==want['fresh'] and raw['pin']==1536 and not raw['stop_sent'],'Exact scheduled native input differs')
  command='GEN 64 fresh=%d pin=1536 ckpt=1 temperature=0 '%want['fresh']+' '.join(map(str,want['ids']))
  require(raw['command']==command,'Exact no-logprobs native command differs')
  meta=numeric.extract_numeric(raw,Path(raw['capture_directory']),True,False,[(0,24),(24,48)])
  ledger=meta['ledger'];require(ledger['actual_reused']==want['expected_reused'] and ledger['evaluated_prompt_rows']==len(want['ids'])-want['expected_reused'],'Actual reuse/new-work differs')
  require(raw['output_ids'] and not ledger['cancelled'],'Completed greedy output required')
  head=next(v for v in meta['vectors']if v['phase']=='first_logits_before_sampler')
  times=raw['stdout_seconds'];first=next(t for t,line in zip(times,raw['stdout'])if line.startswith('T '))
  views.append({'ids':raw['ids'],'output_ids':raw['output_ids'],'head_sha256':head['sha256'],'head_bytes':int(head['bytes']),'actual_reused':ledger['actual_reused'],'new_prompt_rows':ledger['evaluated_prompt_rows'],'first_token_receive_seconds':first})
 return views

def equivalence(a,b):
 require(len(a)==len(b)==6,'Both complete arms required')
 for left,right in zip(a,b):require(all(left[k]==right[k]for k in('ids','output_ids','head_sha256','head_bytes')),'Cold/hit outputs or full head differ')
 return {'bounded_basic_cache_passed':True,'full_cache_qualified':False,'rows':[{'cold_ttft_seconds':x['first_token_receive_seconds'],'cached_ttft_seconds':y['first_token_receive_seconds'],'actual_reused':y['actual_reused'],'new_prompt_rows':y['new_prompt_rows']}for x,y in zip(a,b)]}

NS={**parent.NS}
for name,value in list(NS.items()):
 if isinstance(value,types.FunctionType) and value.__globals__ is parent.NS:NS[name]=types.FunctionType(value.__code__,NS,value.__name__,value.__defaults__,value.__closure__)
NS.update(__file__=__file__,PLAN=HERE/'source52-serial-cache-latency-recipe-v1.json',closure=closure,phase_recipe=phase_recipe,Protocol=Protocol,preflight=preflight)
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
 old=parent.NS;parent.NS=NS
 try:
  NS['_PORTS']=parent.base.boundary_ports(NS)
  if '--construction-only'in rest:
   old_set_case=parent.set_case;old_preflight=parent.preflight
   parent.set_case=NS['set_case'];parent.preflight=preflight
   try:sys.argv=[sys.argv[0],*rest];parent.main()
   finally:parent.set_case=old_set_case;parent.preflight=old_preflight
  else:sys.argv=[sys.argv[0],*rest];parent.main()
 finally:parent.NS=old
if __name__=='__main__':main()
