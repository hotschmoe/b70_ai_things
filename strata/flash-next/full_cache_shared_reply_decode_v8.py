"""Fresh root-owned metadata decode using original tokenizer source, no historical output proof."""
import hashlib,json,os
from pathlib import Path
from full_cache_shared_history_v2 import require,digest,canonical_sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNNER=HERE/'qualify_api_positive_overlap_cpu_screen_v3.py'
FINITE_PLAN=HERE/'api-positive-overlap-cpu-screen-source-plan-v3.json'
CONTAINER_RUNNER='/harness/strata/flash-next/pilot.py'
FINITE_PLAN_SHA='a3de902172c24fb11a7c1c09ac1a096a6e7cdfa671403901ae58dd5c830c6492'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mechanism_binding():
 require(sha(FINITE_PLAN)==FINITE_PLAN_SHA,'Frozen finite metadata source association changed');plan=json.loads(FINITE_PLAN.read_bytes());require(sha(RUNNER)==plan['runner_sha256'],'Original metadata tokenizer mechanism changed')
 return {'original_runner_path':str(RUNNER),'original_runner_sha256':sha(RUNNER),'finite_source_plan_sha256':sha(FINITE_PLAN),'mechanism_only_no_old_output_proof_transfer':True}

def batch_requests(phases,events):
 requests=[];records=[]
 for phase in phases:
  prototype=phase['phase']
  require(len(phase['client']['rows'])==len(prototype['rows'])==len(phase['work']),'Every original phase reply must be independently decoded')
  for index,client in enumerate(phase['client']['rows']):
   messages=prototype['rows'][index]['messages'];begins=[e for e in events if e['kind']=='engine_begin' and e['sequence']>=phase['producer_ack']['sequence'] and e['sequence']<=phase['end_sequence'] and e['rendered_prompt']['messages']==messages];require(len(begins)==1,'Exact per-reply original native begin required');begin=begins[0];ends=[e for e in events if e['kind']=='engine_end' and e['call']==begin['call']];require(len(ends)==1,'Exact per-reply native end required');end=ends[0]
   require(begin['engine_pid']==end['engine_pid'] and begin['engine_generation']==end['engine_generation'] and end['generated_ids_sha256']==digest(end['generated_ids']) and client['request']['messages']==messages and client['error'] is None,'Actual per-reply source/client association differs')
   work=[w for w in phase['work']if w['call']==begin['call']];require(len(work)==1 and work[0]['rid']==end['rid'] and type(end['rid'])is int and end['rid']>0,'Original actual strict RID/call required')
   partial=client['cancel_requested'];require(type(partial)is bool and (partial or client['done_received'] is True) and end['cancelled'] is partial,'Intentional cancellation versus complete native reply differs');text='';done=False
   for item in client['events']:
    data=item['data']
    if data=='[DONE]':require(not done,'Duplicate per-reply SSE terminal');done=True;continue
    require(not done,'Text after SSE terminal');row=json.loads(data);require(not row.get('error') and len(row.get('choices',[]))==1,'Original per-reply SSE error/shape');value=row['choices'][0].get('delta',{}).get('content');require(value is None or type(value)is str,'Original content type');text+=value or ''
   require(done or partial,'Complete or intentionally cancelled original SSE required');request={'case':begin['call'],'repeat':end['rid'],'tokens':list(end['generated_ids']),'content':text};requests.append(request);records.append({'phase':prototype['name'],'client_index':index,'messages':messages,'client':client,'native_begin':begin,'native_end':end,'mode':'intentional_cancel_delivered_text_prefix' if partial else 'complete_API_yielded_ID_text'})
 require(requests and len({(r['case'],r['repeat'])for r in requests})==len(requests),'Complete unique actual reply roster required');return requests,records

def decoded_binding(requests,records,observed,tokenizer_hashes):
 require(observed['actual_GPU_touch'] is False and observed['actual_model_payload_read'] is False and observed['fixtures']==[] and observed['tokenizer_file_sha256']==tokenizer_hashes and len(observed['decoded_outputs'])==len(requests),'New own-tokenizer full reply roster required');result=[]
 for request,source,row in zip(requests,records,observed['decoded_outputs']):
  require(set(row)=={'case','repeat','text','matches'} and row['case']==request['case'] and row['repeat']==request['repeat'] and type(row['text'])is str and type(row['matches'])is bool and row['matches']==(row['text']==request['content']),'Actual own decode request/result association differs')
  partial=source['mode']=='intentional_cancel_delivered_text_prefix';require(row['text'].startswith(request['content']) if partial else row['text']==request['content'],'Independent own API-yielded IDs/text differ');result.append({'phase':source['phase'],'call':request['case'],'actual_RID':request['repeat'],'original_API_yielded_ids':request['tokens'],'decoded_text':row['text'],'delivered_HTTP_text':request['content'],'mode':source['mode'],'cancelled_HTTP_delivered_token_count_observed':False if partial else None})
 return {'actual_all_reply_decodes':result,'original_API_yielded_ID_HTTP_text_qualified':True,'native_accepted_ID_roster_authority_borrowed_from_API_yields':False,'actual_cancelled_HTTP_delivered_token_ID_roster_observed':False,'full_cache_runtime_qualified':False}

def tokenizer_location(actor_plan,prepared,artifact):
 require(actor_plan['pack']==prepared['pack'],'Actual source40 prepared pack association differs')
 require(set(artifact['tokenizer_files'])=={'vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja'},'Exact admitted original tokenizer metadata roster required')
 return Path(prepared['pack'])/'tokenizer',dict(artifact['tokenizer_files'])

from full_cache_shared_metadata_retirement_v8 import retire as retire_owned_metadata

def command_recipe(output,source,tokenizer_pack,image,owner_digest,name,actor_plan_digest):
 """Original inside_tokenizer argv, with a new source/owner input association."""
 output=Path(output).resolve();source=Path(source).resolve();pack=Path(tokenizer_pack).resolve()
 require(image==IMAGE,'Exact original metadata-only tokenizer image required')
 require(type(owner_digest)is str and len(owner_digest)==64 and all(c in '0123456789abcdef' for c in owner_digest),'Actual new decode owner digest required');require(type(name)is str and name.startswith('b70-prefix-') and '-shared40-reply-decode-' in name,'Distinct parent-owned metadata name required');mechanism_binding()
 require(type(actor_plan_digest)is str and len(actor_plan_digest)==64 and all(c in '0123456789abcdef' for c in actor_plan_digest),'Actual parent/actor snapshot digest required')
 mounts=[(RUNNER,CONTAINER_RUNNER,'ro'),(output/'decode-plan.snapshot.json','/pilot-plan.json','ro'),(source/'tools/strata_tokenizer.py','/native-tools/strata_tokenizer.py','ro'),(source/'serve/frontend.py','/native-serve/frontend.py','ro'),(output,'/results','rw')]
 # Mount only declared original tokenizer metadata, never a runtime pack or
 # model directory that happens also to contain the same metadata files.
 mounts += [(pack/n,'/tokenizer/'+n,'ro') for n in ('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')]
 require(all(not p.is_symlink() for p,_,_ in mounts),'Decode mounts cannot be symlinks')
 cmd=['docker','run','--name',name,'--label','b70.shared40.reply.decode='+owner_digest,'--label','b70.prefix.plan='+actor_plan_digest,'--network','none','--read-only','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','128','--user','1000:1000','--entrypoint','/usr/bin/env','-w','/harness']
 for path,dst,mode in mounts:cmd+=['-v',str(path)+':'+dst+':'+mode]
 return cmd+[image,'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python',CONTAINER_RUNNER,'--inside-tokenizer','/pilot-plan.json','--token-output','/results/decoded-output.json','--decode-input','/results/output-id-requests.json']

def owned_recipe(obj,command,owner,actor_digest):
 name=command[command.index('--name')+1];h=obj['HostConfig'];c=obj['Config'];expected=command[command.index(IMAGE)+1:]
 require(obj['Name']=='/'+name and obj['Image']==IMAGE and c['Image']==IMAGE and c['Labels'].get('b70.shared40.reply.decode')==owner and c['Labels'].get('b70.prefix.plan')==actor_digest and c['Entrypoint']==['/usr/bin/env'] and c['Cmd']==expected,'Actual new metadata owner/image/command differs')
 require(c['WorkingDir']=='/harness' and c['User']==command[command.index('--user')+1],'Actual metadata cwd/user differs')
 require(h['Memory']==h['MemorySwap']==512<<20 and h['NanoCpus']==1000000000 and h['PidsLimit']==128 and h['NetworkMode']=='none' and h['ReadonlyRootfs'] is True and not h.get('Devices') and not h.get('DeviceRequests') and not h.get('GroupAdd') and not h.get('Privileged'),'Actual metadata CPU-only bounded recipe differs')
 binds=[command[i+1] for i,arg in enumerate(command) if arg=='-v'];require(h['Binds']==binds,'Actual metadata mount recipe differs')
 declared=sorted((src,dst,mode=='rw','bind') for src,dst,mode in (b.rsplit(':',2) for b in binds));actual=sorted((m['Source'],m['Destination'],m['RW'],m['Type']) for m in obj['Mounts']);require(declared==actual,'Actual metadata has missing or extra mounts');return True

def execute(actor_plan,actor_root,output,phases,events):
 """ROOT ONLY: new CPU metadata job while the same root owns the actor lease."""
 import full_cache_shared_runtime_v8 as ctrl
 from finite_tokenizer_owned_execution_v2 import execute as owned_execute
 before=ctrl.manifest_binding(actor_plan);ctrl.c1.leased([0,1]);actor_root=Path(actor_root);require(ctrl.canonical(ctrl.read(actor_root/'plan.snapshot.json'))==ctrl.canonical(actor_plan),'Actual actor snapshot differs');launch=ctrl.read(actor_root/'launch.command.json');require(launch[launch.index('--name')+1].startswith('b70-prefix-'+str(os.getpid())+'-'),'Decode must execute in its own actual actor controller');actor_digest=sha(actor_root/'plan.snapshot.json');output=Path(output);require(not output.exists(),'Fresh own-history decode directory required');output.mkdir(parents=True)
 requests,records=batch_requests(phases,events)
 prepared=ctrl.read(Path(actor_plan['prepared'])/'prepared.json');source=Path(prepared['engine_receipt']).parent/'source';pack,expected_hashes=tokenizer_location(actor_plan,prepared,ctrl.read(Path(actor_plan['prepared'])/'artifact-identity.json'));require(sha(prepared['pack_receipt'])==prepared['pack_receipt_sha256'],'Actual pack metadata receipt changed')
 hashes={name:sha(pack/name) for name in ('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')}
 require(hashes==expected_hashes,'Current explicit prepared tokenizer metadata differs');packet={'corpus_messages':[],'own_requests':requests,'own_source_records':records,'tokenizer_root':str(pack),'pack_receipt_sha256':prepared['pack_receipt_sha256'],'tokenizer_file_sha256':hashes,'current_source40_binding':before,'mechanism':mechanism_binding(),'old_response_or_state_proof_transferred':False}
 def write(path,row):Path(path).write_text(json.dumps(row,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
 write(output/'decode-plan.snapshot.json',packet);write(output/'output-id-requests.json',requests);owner=sha(output/'decode-plan.snapshot.json');name='b70-prefix-'+str(os.getpid())+'-shared40-reply-decode-'+owner[:12]
 command=command_recipe(output,source,pack,IMAGE,owner,name,actor_digest);write(output/'command.json',command)
 def owned(obj):return owned_recipe(obj,command,owner,actor_digest)
 import subprocess
 import signal,time
 from observe_cpu_swap_attribution_v4 import inspect
 ledger={'schema':2,'name':name,'command':command,'actor_controller_pid':os.getpid(),'actor_plan_sha256':actor_digest,'metadata_input_sha256':owner,'original_launch_subreaper_and_session_retirement_required':True,'actual_owned_container_observed':False,'forced_owned_stop':False,'owned_terminal_and_removed':False,'retirement_failures':[],'launch_error':None,'interruption_signals':[]};write(output/'metadata-ownership-ledger.json',ledger)
 previous={};receipt=None;obj=None
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous[sig]=signal.signal(sig,lambda number,frame:ledger['interruption_signals'].append(number))
 try:
  receipt=owned_execute(command,output,owned,120);write(output/'owned-tokenizer-receipt.json',receipt)
 except BaseException as exc:ledger['launch_error']=type(exc).__name__+': '+str(exc);raise
 finally:
  # owned_execute keeps the original CLI/session until creation/launch settles,
  # even on errors. This separate census retires its exact declared container.
  try:obj=retire_owned_metadata(name,owned,output,ledger,inspect,subprocess.run,time.sleep);write(output/'owned-terminal-inspection.json',obj)
  finally:
   for sig,handler in previous.items():signal.signal(sig,handler)
 state=obj['State'];require(ledger['launch_error'] is None and ledger['interruption_signals']==[] and ledger['retirement_failures']==[] and ledger['forced_owned_stop'] is False,'Metadata recovery/interruption does not qualify a normal original decode')
 require(receipt['return_code']==0 and receipt['error'] is None and receipt['stop_signals']==[] and receipt['launch_descendants']['launch_session_empty'] is True and receipt['launch_descendants']['adopted_children']=={} and receipt['launch_descendants']['tracking_errors']==[] and state['ExitCode']==0 and not state['OOMKilled'] and not state['Error'],'Actual owned independent metadata decode failed')
 observed=ctrl.read(output/'decoded-output.json');derivation=decoded_binding(requests,records,observed,hashes)
 require(ctrl.canonical(ctrl.manifest_binding(actor_plan))==ctrl.canonical(before) and hashes=={name:sha(pack/name) for name in hashes},'Current source/tokenizer changed during own history decode')
 require(ctrl.read(output/'decode-plan.snapshot.json')==packet and ctrl.read(output/'output-id-requests.json')==requests and sha(output/'decode-plan.snapshot.json')==owner,'Original owned decode inputs changed')
 result={'derivation':derivation,'actual_owned_decode_receipt':receipt,'decode_receipt_sha256':sha(output/'owned-tokenizer-receipt.json'),'decoded_output_sha256':sha(output/'decoded-output.json'),'current_source40_binding':before,'full_cache_runtime_qualified':False}
 write(output/'reply-decode-binding.json',result);return result

def finalized_binding(actor_parent,decode_root):
 """ROOT READONLY: repeat current closed source and actual own-record joins."""
 import full_cache_shared_runtime_v8 as ctrl
 import full_cache_shared_closed_source_v8 as closed
 from serial37_canonical_json_v3 import read_unique
 actor_parent=Path(actor_parent).resolve();child_root=actor_parent/'child';decode_root=Path(decode_root).resolve();require(decode_root==child_root/'own-reply-decode' and not decode_root.is_symlink(),'Exact confined own-history decode directory required')
 _,plan,child=closed.admit(actor_parent,ctrl,HERE/'qualify_full_cache_shared_runtime_v8.py');before=ctrl.manifest_binding(plan)
 read=read_unique;packet=read(decode_root/'decode-plan.snapshot.json');saved=read(decode_root/'reply-decode-binding.json');receipt=read(decode_root/'owned-tokenizer-receipt.json');ledger=read(decode_root/'metadata-ownership-ledger.json');obj=read(decode_root/'owned-terminal-inspection.json')
 actual=[json.loads(line) for line in (child_root/'api-native-trace.jsonl').read_bytes().splitlines()];requests,records=batch_requests(child['phase_receipts'],actual);require(requests==packet['own_requests'] and records==packet['own_source_records'] and read(decode_root/'output-id-requests.json')==requests,'Every original phase/reply decode input/source row changed')
 prepared=read(Path(plan['prepared'])/'prepared.json');tokenizer,hashes=tokenizer_location(plan,prepared,read(Path(plan['prepared'])/'artifact-identity.json'));require(packet['tokenizer_root']==str(tokenizer) and packet['pack_receipt_sha256']==prepared['pack_receipt_sha256']==sha(prepared['pack_receipt']) and packet['tokenizer_file_sha256']==hashes=={n:sha(tokenizer/n) for n in hashes},'Actual current tokenizer/pack receipt association changed')
 require(packet['corpus_messages']==[] and packet['mechanism']==mechanism_binding() and packet['old_response_or_state_proof_transferred'] is False and ctrl.canonical(packet['current_source40_binding'])==ctrl.canonical(before),'Own new metadata purpose/current source changed')
 owner=sha(decode_root/'decode-plan.snapshot.json');name='b70-prefix-'+str(child['producer_pid'])+'-shared40-reply-decode-'+owner[:12];command=command_recipe(decode_root,Path(prepared['source']),tokenizer,IMAGE,owner,name,child['plan_sha256']);require(read(decode_root/'command.json')==ledger['command']==receipt['command']==command and ledger['name']==name and ledger['actor_controller_pid']==child['producer_pid'] and ledger['actor_plan_sha256']==child['plan_sha256'] and ledger['metadata_input_sha256']==owner,'Actual original metadata command/parent ownership changed');owned_recipe(obj,command,owner,child['plan_sha256'])
 require(ledger['actual_owned_container_observed'] is True and ledger['owned_terminal_and_removed'] is True and ledger['forced_owned_stop'] is False and ledger['retirement_failures']==[] and ledger['launch_error'] is None and ledger['interruption_signals']==[] and ledger['terminal_inspection']==obj,'Actual normal owned metadata retirement changed')
 state=obj['State'];launch=receipt['launch_descendants'];require(type(receipt['return_code'])is int and receipt['return_code']==0 and receipt['error'] is None and receipt['stop_signals']==[] and receipt['subreaper']=={'subreaper':True,'owner_pid':child['producer_pid']} and receipt['client_pid']==launch['producer_pid']==launch['producer_session'] and launch['launch_session_empty'] is True and launch['adopted_children']=={} and launch['tracking_errors']==[] and type(state['ExitCode'])is int and state['ExitCode']==0 and state['Running'] is False and state['OOMKilled'] is False and state['Error']=='','Actual original metadata CLI/session/terminal failed')
 observed=read(decode_root/'decoded-output.json');derivation=decoded_binding(requests,records,observed,hashes);require(derivation==saved['derivation'] and receipt==saved['actual_owned_decode_receipt'] and saved['decode_receipt_sha256']==sha(decode_root/'owned-tokenizer-receipt.json') and saved['decoded_output_sha256']==sha(decode_root/'decoded-output.json') and ctrl.canonical(saved['current_source40_binding'])==ctrl.canonical(before) and saved['full_cache_runtime_qualified'] is False,'Actual own decode result/receipt source changed')
 require(ctrl.canonical(ctrl.manifest_binding(plan))==ctrl.canonical(before),'Current source40 changed during own decode recollection');closed.admit(actor_parent,ctrl,HERE/'qualify_full_cache_shared_runtime_v8.py');return saved
