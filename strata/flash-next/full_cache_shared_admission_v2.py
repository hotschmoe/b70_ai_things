"""Source-only authentic sharedpin case admission; no cache/math/runtime authority."""
import copy,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SEED=HERE/'full-cache-shared-case-seed-v2.json';SOURCE_PLAN=HERE/'full-cache-shared-source-plan-v2.json'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl')
TOKENIZER=Path('/mnt/vm_8tb/b70/models/flashnext-native-source-pack-20261009-v3/tokenizer')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token_sha(ids):return hashlib.sha256(b''.join(int(t).to_bytes(4,'little') for t in ids)).hexdigest()

def fixture_command(directory,container):
 require(re.fullmatch(r'b70-full-cache-tokenizer-v2-\d+',container) is not None,'Owned CPU fixture container name required');directory=Path(directory).resolve();producer=HERE/'render_full_cache_fixture_v2.py'
 command=['docker','run','--name',container,'--label','b70.tokenizer.fixture='+sha(producer),'--network','none','--read-only','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','128','--user','1000:1000','--entrypoint','/usr/bin/env']
 for source,target,mode in [(producer,'/harness/render_full_cache_fixture_v2.py','ro'),(SEED,'/seed.json','ro'),(SDK/'source/tools/strata_tokenizer.py','/tools/strata_tokenizer.py','ro'),(SDK/'source/serve/frontend.py','/serve/frontend.py','ro'),(TOKENIZER,'/tokenizer','ro'),(directory,'/results','rw')]:command+=['-v',str(source)+':'+target+':'+mode]
 return command+[IMAGE,'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python','/harness/render_full_cache_fixture_v2.py']

def fixture_admission(directory):
 directory=Path(directory).resolve();data=read(directory/'fixtures.json');seed=read(SEED);receipt=read(directory/'receipt.json');state=receipt['state'];require(receipt['return_code']==0 and state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'] and receipt['removed'] is True and receipt['devices']==[] and receipt['device_requests'] is None and receipt['image']==IMAGE,'Actual CPU-only tokenizer image/terminal/noGPU receipt required')
 require(data['source_seed_sha256']==sha(SEED)==sha(directory/'seed.snapshot.json') and read(directory/'seed.snapshot.json')==seed and data['producer_source_sha256']==sha(HERE/'render_full_cache_fixture_v2.py'),'Exact authentic literal-role/template seed/producer bytes required')
 require(data['actual_GPU_touch'] is False and data['actual_model_payload_read'] is False and data['shared_cache_runtime_qualified'] is False,'Tokenization cannot transfer native cache/runtime proof')
 command=read(directory/'command.json');require(command[:2]==['docker','run'] and command.count(IMAGE)==1 and '--device' not in command and '--gpus' not in command and not any('.gguf' in w for w in command),'Actual noGPU/noGGUF tokenizer recipe required');require(command[command.index('--network')+1]=='none' and '--read-only' in command,'Tokenizer isolation required')
 require(command==fixture_command(directory,command[command.index('--name')+1]),'Exact reviewed CPU fixture fullcommand/mount/producer association differs')
 mounts=[command[i+1] for i,w in enumerate(command) if w=='-v'];require(len(mounts)==6 and all(':/model' not in w and ':/pack' not in w for w in mounts),'Only explicit seed/producer/tokenizer/tools/frontend/results mounts allowed')
 expected={('/'+n):v for n,v in read(HERE/'current-ple-prompt35-engine-build-plan-v1.json')['expected_patched_source_sha256'].items() if n in ('tools/strata_tokenizer.py','serve/frontend.py')};require(data['source_file_sha256']==expected and data['tokenizer_file_sha256']==read(HERE/'batch-numerical-case2-source35-v2.json')['tokenizer_sha256'],'Original source35 tokenizer/frontend/packtokenizer identity differs')
 require(set(data['fixtures'])==set(seed['messages']),'Exact declared phase roster required')
 for phase,messages in seed['messages'].items():
  rows=data['fixtures'][phase];require(len(rows)==len(messages),'Exact phase count required')
  for index,(row,message) in enumerate(zip(rows,messages)):
   require(row['logical_index']==index and row['messages']==message and 0<len(row['ids'])<1984 and all(type(t) is int and 0<=t<248320 for t in row['ids']),'Exact actual literal-role/tokenized fixture row required')
 return data,{'directory':str(directory),'evidence_sha256':{n:sha(directory/n) for n in ('fixtures.json','receipt.json','command.json','seed.snapshot.json')},'producer_source_sha256':sha(HERE/'render_full_cache_fixture_v2.py'),'actual_CPU_tokenizer_only':True,'shared_cache_runtime_qualified':False}

def shared_boundary(fixtures):
 phases=('prime_shared','target_shared','prefill_cancel','decode_cancel');rows=[(phase,r) for phase in phases for r in fixtures[phase]];first=rows[0][1]['ids'];turns=[i for i,t in enumerate(first) if t==248045];require(len(turns)==3 and turns[0]==0,'Exact first system/user/assistant boundaries required');n=turns[1];require(0<n<len(first),'Valid exact shared prefix boundary required');prefix=first[:n]
 for phase,row in rows:
  ids=row['ids'];positions=[i for i,t in enumerate(ids) if t==248045];require(len(positions)==3 and positions[1]==n and ids[:n]==prefix and n<len(ids),'Every declared shared/divergent owner musthave identical actual acceptedprefix')
 require(len({tuple(r['ids'][n:]) for r in fixtures['target_shared']})==2,'Actual divergent target suffixes required')
 require([r['ids'] for r in fixtures['fresh_shared']]==[r['ids'] for r in fixtures['target_shared']],'Actual freshcontrols mustuse identical whole target token inputs, pinABSENT')
 for row in fixtures['independent']+fixtures['eviction_unpinned']:
  require(row['ids'][:n]!=prefix,'Independent/eviction owner cannot masquerade as shared prefix')
 return {'tokens':n,'accepted_prefix_convention':'ids[0:N), excludes user turn-token at indexN','sha256_le32':token_sha(prefix),'ids':prefix,'runtime_pin_creation_and_reuse_still_required':True}

def source_support_binding(generate_path,public_header_path):
 expected=read(HERE/'current-ple-prompt35-engine-build-plan-v1.json')['expected_patched_source_sha256'];require(sha(generate_path)==expected['sycl/src/program/generate.cpp'] and sha(public_header_path)==expected['sycl/include/strata/core/batch_public_prefix.hpp'],'Exact consumed source35 request guard/header bytes required')
 generate=Path(generate_path).read_text();header=Path(public_header_path).read_text();require('return fresh?0:(pin_present?pin:prompt-1);' in header and 'const bool public_batch_request=batch_public_enabled && strict_batch_identity;' in generate,'Actual independently implemented publicfresh/pin selection mustbe inspected')
 return {'startup_hint_preserved':'public_pin_fresh=unsupported','startup_hint_is_runtime_support_authority':False,'combinedfreshpin_source_implemented':True,'combinedfreshpin_runtime_qualified':False,'conservative_case_combination_admitted':False,'scope':'exact source35 parser/ceiling and pin save implementation, not a runtimequalification or staleINFO inference'}

def body_policy(fresh,pin=None):
 require(type(fresh) is bool,'Actual fresh bool required');body={'strata_fresh':fresh}
 if pin is not None:require(not fresh,'Combinedfresh+pin is sourceimplemented but outside this admittedcase runtime scope');require(type(pin) is int and pin>=0,'Exact integer shared pin required');body['strata_shared_prefix']={'tokens':pin}
 return body

def cache_work(prompt,pin,fresh,reused,read_from,reread_to,evaluated_ids):
 require(type(pin) is int and 0<=pin<len(prompt) and type(fresh) is bool,'Source publicpin/fresh domain invalid');expected=0 if fresh else pin;require(reused==read_from==expected and reread_to==-1 and evaluated_ids==prompt[expected:],'Actual declared reuse/new fullprompt work differs')
 return {'actual_reused':expected,'actual_new_prompt_tokens':len(prompt)-expected,'actual_read_from':expected,'actual_reread_to':-1,'cache_math_qualified':False}

def generate(directory,output):
 data,binding=fixture_admission(directory);boundary=shared_boundary(data['fixtures']);seed=read(SEED);n=boundary['tokens'];case={'schema':'full-cache-shared-CPU-case-v2','source_lane':'source35','slots':2,'authentic_fixture_binding':binding,'shared_boundary':boundary,'phases':{},'genuine_runtime_preparable':False,'full_cache_runtime_qualified':False,'coldprime_actualzero_counter_prerequisite':True,'combinedfreshpin_sourceimplemented_not_admitted':True,'full_model_math_qualified':False,'latency_qualified':False}
 for phase,rows in data['fixtures'].items():
  shared=phase in ('prime_shared','target_shared','prefill_cancel','decode_cancel');fresh=phase in ('warm','fresh_shared');case['phases'][phase]={'rows':rows,'actual_HTTP_body_policy':body_policy(fresh,n if shared else None),'native_keys_required':['fresh='+str(int(fresh))]+(['pin='+str(n)] if shared else []),'expected_initial_reused':0 if fresh or not shared or phase=='prime_shared' else n,'expected_new_prompt_tokens_by_row':[len(r['ids'])-(0 if fresh or not shared or phase=='prime_shared' else n) for r in rows],'actual_source_counter_readback_still_required':True}
 require(not Path(output).exists(),'NEW authenticated case output cannot overwrite evidence');Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(case,indent=2,ensure_ascii=True)+'\n',encoding='ascii');return case
