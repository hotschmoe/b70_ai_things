"""Root-owned concrete shared-cache HTTP phase execution; no full-cache PASS."""
import json,os,re,signal,subprocess,sys,threading,time,traceback,urllib.request,urllib.error
from pathlib import Path
import c1_serve_controller_combined_v140_v3 as c1
from full_cache_shared_http_client_v2 import cohort
from full_cache_shared_phase_contract_v2 import calls_and_work,phase_events,full_scope_status
from full_cache_shared_raw49_v2 import collect
from full_cache_shared_terminal_v2 import associate_events
from full_cache_shared_prefix_jobs_v2 import prefix_jobs
from full_cache_shared_native_lifetime_v2 import recollect as native_lifetime
from batch_numerical_proofs_v40 import owner_proofs
ABORT=threading.Event()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,row):Path(path).write_text(json.dumps(row,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
def sha(path):
 import hashlib
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def events(path):
 if not Path(path).exists():return []
 raw=Path(path).read_bytes();require(not raw or raw.endswith(b'\n'),'Producer trace not fully flushed/newline terminated');return [json.loads(line) for line in raw.splitlines() if line]
def command_recipe(plan,out,pid):
 out=Path(out).resolve();root=Path(__file__).resolve().parents[2];here=Path(__file__).resolve().parent;prepared=read(Path(plan['prepared'])/'prepared.json');engine=Path(prepared['engine_receipt']).parent;lock=read(here/'model-lock.json');name='b70-prefix-'+str(pid)+'-fullcache-shared40-v3'
 command=['docker','run','-d','--name',name,'--label','b70.fullcache.shared.plan='+sha(out/'plan.snapshot.json'),'--label','b70.prefix.plan='+sha(out/'plan.snapshot.json'),'--entrypoint','/bin/bash','--network','host','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--group-add',str(os.stat('/dev/dri/card0').st_gid)]
 for src,dst,ro in [(engine/'build','/build',True),(engine/'source','/src',True),(Path(plan['pack']),'/pack',True),(root/lock['destination'],'/model',True),(out,'/results',False),(here,'/controller',True)]:command+=['-v',str(src)+':'+dst+(':ro' if ro else ':rw')]
 env={'B70_BATCH_TRACE':'/results/api-native-trace.jsonl','B70_BATCH_NATIVE_LOG':'/results/engine.combined.log','B70_BATCH_ARM_REQUEST':'/results/ARM.request','B70_BATCH_ARM':'/results/ARM','B70_FULLCACHE_PHASE_REQUEST':'/results/phase.request.json','B70_FULLCACHE_PHASE_ACK':'/results/phase.ack.json','B70_FULLCACHE_CONTROL_REQUEST':'/results/control.request.json','B70_FULLCACHE_CONTROL_ACK':'/results/control.ack.json','STRATA_ARTIFACT_IDENTITY_SHA256':sha(out/'artifact-identity.json')}
 for k,v in sorted(env.items()):command+=['-e',k+'='+v]
 return command+[plan['image'],'-lc','set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec /opt/b70-c1-python/bin/python /controller/full_cache_shared_api_trace_v4.py --engine strata --config /results/server-config.json --host 127.0.0.1 --port '+str(plan['port'])]
def owned(plan,out,pid):
 name='b70-prefix-'+str(pid)+'-fullcache-shared40-v3';obj=c1.inspected(name);require(obj['Name']=='/'+name and obj['Config']['Image']==plan['image'] and obj['Config']['Labels'].get('b70.fullcache.shared.plan')==sha(Path(out)/'plan.snapshot.json'),'Actual foreign full-cache actor owner/image/label')
 from full_cache_shared_memory_capture_v4 import recipe_binding
 recipe_binding(obj,command_recipe(plan,out,pid),plan['image']);return obj
def publish_phase(out,index,name,deadline=30,render_request=None):
 out=Path(out);path=out/'phase.request.json';temporary=out/'phase.request.tmp';packet={'index':index,'name':name}
 if render_request is not None:packet['render_request']=render_request
 write(temporary,packet);temporary.replace(path);limit=time.monotonic()+deadline
 while True:
  if (out/'phase.ack.json').exists():
   row=read(out/'phase.ack.json')
   if row.get('index')==index and row.get('name')==name:require(type(row['sequence'])is int and row['sequence']>0 and type(row['engine_pid'])is int and row['engine_pid']>0,'Actual phase acknowledgement lacks native process');return row
  require(time.monotonic()<limit,'Owned producer phase publication timeout');time.sleep(.025)
def phase_trace(trace,name,index):
 lines=trace.splitlines();mark='HARNESS FULLCACHE_PHASE index='+str(index)+' name='+name;matches=[i for i,line in enumerate(lines) if line==mark];require(len(matches)==1,'Exact unique actual producer phase marker required');start=matches[0];end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('HARNESS FULLCACHE_PHASE ')),len(lines));return '\n'.join(lines[start+1:end]),{'original_first_line':start+2,'original_last_line':end,'marker':mark}
def cancellation_watch(out,phase,ack,gate,stop,receipt):
 """Observed progress/body gate only; never wallclock cancellation authority."""
 while not stop.wait(.025):
  seen=[e for e in events(Path(out)/'api-native-trace.jsonl') if e['sequence']>ack['sequence']];begins=[e for e in seen if e['kind']=='engine_begin']
  if phase['cancel_kind']=='prefill':
   sends=[e for e in seen if e['kind']=='native_send' and e['line'].startswith(('GEN ','BGEN '))];progress=[e for e in seen if e['kind']=='native_receive' and e['line'].startswith('PP ')]
   if not progress:continue
   require(len(begins)==len(sends)==1 and begins[0]['engine_pid']==ack['engine_pid']==sends[0]['engine_pid'],'Prefill PP must have sole actual owned pending admission');p=progress[0];require(p['engine_pid']==ack['engine_pid'] and sends[0]['sequence']<p['sequence'] and not any(e['kind']=='native_receive' and e['line'].startswith(('T ','BT ','DONE ','BADM ','BDONE ')) for e in seen),'Actual prefill cancellation gate reached after output/terminal');values=p['line'].split();require(0<int(values[1])<len(phase['rows'][0]['ids']),'Actual PP must precede whole prompt completion');receipt.update(kind='actual_owned_PP_before_first_token',source_record=p,sole_native_command=sends[0],engine_begin=begins[0]);write(gate,receipt);return
  body=[e for e in seen if e['kind']=='native_receive' and e['line'].startswith('SBF batch_event ') and dict(re.findall(r'(\w+)=([^ ]+)',e['line'])).get('rows')=='2']
  if body:receipt.update(kind='actual_completed_two_row_decode',source_record=body[0]);write(gate,receipt);return
def stale_watch(out,ack,stop,receipt):
 """Wait for actual continued new owner, not an assumed slot or a timer."""
 from full_cache_shared_stale_owner_v2 import owner_rows,command as stale_command
 try:
  while not stop.wait(.025):
   owners=owner_rows(events(Path(out)/'api-native-trace.jsonl'));current=[r for r in owners if r['send_sequence']>ack['sequence'] and r['admitted'] and not r['terminal']];old=[r for r in owners if r['terminal'] and r['terminal_sequence']<ack['sequence']]
   pairs=[(a,b) for a in old for b in current if a['pid']==b['pid'] and a['engine_generation']==b['engine_generation'] and a['slot']==b['slot'] and a['rid']!=b['rid'] and a['slotgen']<b['slotgen']]
   if not pairs:continue
   pair=min(pairs,key=lambda pair:(pair[1]['slot'],-pair[0]['slotgen']));proposal=stale_command(*pair);packet={'index':1,'kind':'stale_BSTOP','proposal':proposal};temporary=Path(out)/'control.request.tmp';write(temporary,packet);temporary.replace(Path(out)/'control.request.json');receipt.update(proposal=proposal,request_packet=packet)
   while not stop.wait(.025):
    path=Path(out)/'control.ack.json'
    if path.exists():
     actual=read(path);require(actual=={'index':1,'kind':'stale_BSTOP','proposal':proposal,'engine_pid':proposal['engine_pid']},'Actual same-actor stale control ACK differs');receipt['producer_ack']=actual;return
   raise ValueError('Actual current request retired before stale control ACK')
  raise ValueError('No real continued new owner available for stale control')
 except BaseException as exc:receipt['error']=type(exc).__name__+': '+str(exc)
def run_phase(plan,phase,out,index,url,acknowledged=None):
 directory=Path(out)/'phases'/phase['name'];directory.mkdir(parents=True,exist_ok=False);ack=publish_phase(out,index,phase['name']) if acknowledged is None else acknowledged;require(ack['index']==index and ack['name']==phase['name'],'Own actual phase ACK differs');write(directory/'phase.recipe.json',phase);write(directory/'producer-ack.json',ack);gate=directory/'cancel.allowed.json';stop=threading.Event();gate_receipt={};thread=None;stale_receipt={};stale_thread=None
 if phase['cancel_kind']:thread=threading.Thread(target=cancellation_watch,args=(out,phase,ack,gate,stop,gate_receipt),daemon=True);thread.start()
 if phase.get('stale_owner_control_required'):stale_thread=threading.Thread(target=stale_watch,args=(out,ack,stop,stale_receipt),daemon=True);stale_thread.start()
 try:client=cohort(url,'hotschmoe-dd',[r['messages'] for r in phase['rows']],directory/'client',cancel_index=phase['cancel_index'],max_new=phase['max_new'],cancel_gate=gate if phase['cancel_kind'] else None,abort_event=ABORT,request_policy=phase['policy'])
 finally:stop.set();thread and thread.join(timeout=5);stale_thread and stale_thread.join(timeout=5)
 require(client['client_transport_completed'],'Actual shared phase HTTP transport incomplete');limit=time.monotonic()+60
 while True:
  all_events=events(Path(out)/'api-native-trace.jsonl');current=[e for e in all_events if e['sequence']>ack['sequence']];calls={e['call'] for e in current if e['kind']=='engine_begin'}
  try:terminals=associate_events(current,calls);require(len(terminals)==len(phase['rows']),'Actual phase native/API end roster missing');lifetimes=native_lifetime(current);break
  except (KeyError,ValueError):require(time.monotonic()<limit,'Actual producer terminals/drain failed');time.sleep(.05)
 # FC39 observations follow the flushed terminal print in actual source40;
 # retain their original receive sequence even if it follows API engine_end.
 last=max(e['sequence'] for e in current);current=phase_events(all_events,ack['sequence'],last);work=calls_and_work(phase,current);trace,bounds=phase_trace((Path(out)/'engine.combined.log').read_text(),phase['name'],index)
 stale_binding=None
 if phase.get('stale_owner_control_required'):
  from full_cache_shared_stale_owner_v2 import recollect as stale_recollect
  require(stale_thread is not None and not stale_thread.is_alive() and stale_receipt.get('error') is None and 'producer_ack' in stale_receipt,'Actual source-bound stale control/ACK missing')
  proposal=stale_receipt['proposal'];sends=[e for e in current if e['kind']=='fullcache_control_send' and e['line']==proposal['command']];require(len(sends)==1,'Unique actual same-actor stale send required');ends=[t for t in terminals.values() if t['rid']==proposal['current_owner']['rid']];require(len(ends)==1,'Actual new owner terminal missing');stale_binding=stale_recollect(proposal,sends[0],current,ends[0]);write(directory/'stale-control.json',{'request_and_ack':stale_receipt,'binding':stale_binding})
 if phase['cancel_kind']:
  require(gate.exists() and thread is not None and not thread.is_alive() and gate_receipt==read(gate),'Real native observation cancellation gate missing');require(sum(e['actual_client_cancelled'] for e in terminals.values())==1,'Actual single client/native cancellation required')
  if phase['cancel_kind']=='prefill':require(next(iter(terminals.values()))['actual_generated_ids']==[],'Prefill cancellation produced a token; decode cancel cannot substitute')
 if phase['native_multirow_required']:require(any(line.startswith('SBF batch_event ') and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rows')=='2' for line in trace.splitlines()),'Actual phase lacks genuine two-row body')
 if phase['name']=='prime0':require(any(line.startswith('BCPUBLIC pin_saved ') and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rid')==str(work[0]['rid']) and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('tokens')=='272' for line in trace.splitlines()),'Actual cold prime failed to save exact public boundary272')
 from full_cache_shared_capture_roster_v2 import observe_phase_rids
 from full_cache_shared_metadata_v2 import records as metadata_records,request_binding
 observe_phase_rids(phase,work)
 observed=metadata_records(trace,ack['engine_pid'])
 main_body=[request_binding(observed,r['native_first_command']['line'],plan['scenario_binding']['selected_capture_pass']['selected_actual_RIDs']) for r in work]
 required=__import__('full_cache_shared_raw49_v2').required_roles(phase,work,terminals);raw=None
 if phase['raw_capture_requested'] and phase['cancel_kind']!='prefill':raw=collect(trace,current,work,[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],Path(out)/'captures',required)
 record={'phase':phase,'producer_ack':ack,'end_sequence':last,'original_line_bounds':bounds,'work':work,'main_body_metadata':main_body,'native_lifetime_binding':lifetimes,'terminal_associations':terminals,'client':client,'raw49':raw,'cancel_gate':gate_receipt or None,'stale_owner_control':stale_binding,'raw49_unobserved_reason':'unselected capture phase or partial-prefill cancellation has no completed final49' if raw is None else None,'fresh49_comparison_completed':False,'full_cache_runtime_qualified':False};write(directory/'receipt.json',record);return record
def run(plan,out,pre_health,admitted_plan_bytes,handoff,post_ack_seal):
 import full_cache_shared_runtime_v4 as ctrl
 ctrl.source_binding();c1.leased([0,1]);health=read(pre_health);require(health['passed'] is True and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh owned parent pairhealth required');out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir();__import__('full_cache_shared_plan_snapshot_v4').write_runtime(out/'plan.snapshot.json',plan,admitted_plan_bytes);prepared=Path(plan['prepared']);cfg=read(prepared/'server-config.json');manifest=read(prepared/'artifact-identity.json');cfg.update(args=plan['args'],env=plan['env'],parallel=2,port=plan['port'],model_name='hotschmoe-dd',aliases=[plan['research_alias']],log='/results/server-engine.log',slot_save_path='/results/sessions');manifest.update(primary_model_name='hotschmoe-dd',research_alias=plan['research_alias']);manifest['runtime'].update(args=plan['args'],env=plan['env']);write(out/'server-config.json',cfg);write(out/'artifact-identity.json',manifest);command=command_recipe(plan,out,os.getpid());write(out/'launch.command.json',command);result={'health_handoff':handoff,'post_ACK_byte_seal':post_ack_seal,'passed':False,'phase_receipts':[],'owned_host_memory_samples':[],'error':None,'started_epoch':time.time(),'full_cache_runtime_qualified':False};started=False;removed=False;state=None
 ABORT.clear();previous_signals={}
 def interrupted(number,frame):ABORT.set();raise KeyboardInterrupt('Actual full-cache actor interrupted '+str(number))
 for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous_signals[number]=signal.signal(number,interrupted)
 try:
  from finite_tokenizer_owned_execution_v2 import execute as settle_owned_launch
  from full_cache_shared_memory_capture_v4 import recipe_binding
  launch_root=out/'owned-launch-client';launch_root.mkdir();started=True
  def actual_actor_owner(obj):
   require(obj['Name']=='/'+command[command.index('--name')+1] and obj['Config']['Labels'].get('b70.prefix.plan')==sha(out/'plan.snapshot.json'),'Actual actor launch owner differs');return recipe_binding(obj,command,plan['image'])
  launch_receipt=settle_owned_launch(command,launch_root,actual_actor_owner,timeout=60);write(launch_root/'receipt.json',launch_receipt);result['owned_launch_client']=launch_receipt
  require(launch_receipt['return_code']==0 and launch_receipt['error'] is None and launch_receipt['stop_signals']==[],'Original actor Docker creation/client did not finish normally')
  url='http://127.0.0.1:'+str(plan['port']);limit=time.monotonic()+900
  while True:
   state=owned(plan,out,os.getpid())['State'];require(state['Running'],'Owned fullcache actor exited before readiness')
   try:
    with urllib.request.urlopen(url+'/v1/models',timeout=5) as f:models=json.load(f)
    require([r['id'] for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==sha(out/'artifact-identity.json') for r in models['data']),'Actual live fullcache served identity differs');write(out/'models.json',models);break
   except (OSError,TimeoutError):require(time.monotonic()<limit,'Fullcache readiness deadline');time.sleep(1)
  history=None;history_prior=None
  for index,prototype in enumerate(plan['schedule'],1):
   phase=prototype;acknowledged=None
   if prototype.get('dynamic_own_history_required'):
    from full_cache_shared_history_v2 import canonical_sha,render_binding,resolve_history
    from full_cache_shared_history_decode_v4 import execute as decode_history
    if history is None:
     prior=next(r for r in result['phase_receipts'] if r['phase']['name']=='prime1');actual=events(out/'api-native-trace.jsonl');call=prior['work'][0]['call'];begin=next(e for e in actual if e['kind']=='engine_begin' and e['call']==call);end=next(e for e in actual if e['kind']=='engine_end' and e['call']==call)
     history=decode_history(plan,out,out/'own-history-decode',prior['phase']['rows'][0]['messages'],prior['client']['rows'][0],begin,end);original_ids=prior['phase']['rows'][0]['ids']
    derivation=history['derivation'];request={'messages':derivation['messages'],'messages_sha256':canonical_sha(derivation['messages']),'template_kwargs':{'enable_thinking':False}};acknowledged=publish_phase(out,index,prototype['name'],render_request=request);rendered=render_binding(request,acknowledged['render_observation'],derivation['engine_pid'])
    phase=resolve_history(prototype,derivation,rendered,original_ids,history_prior['inventory'] if history_prior else None,history_prior['rid'] if history_prior else None)
   from full_cache_shared_memory_capture_v4 import capture as memory_capture
   for boundary in ('before',):
    sample=memory_capture(owned(plan,out,os.getpid()),command,plan,out/'memory-samples'/('phase-'+str(index)+'-'+boundary));sample['phase_index']=index;sample['boundary']=boundary;write(out/'memory-samples'/('phase-'+str(index)+'-'+boundary)/'sample.json',sample);result['owned_host_memory_samples'].append(sample)
   row=run_phase(plan,phase,out,index,url,acknowledged)
   sample=memory_capture(owned(plan,out,os.getpid()),command,plan,out/'memory-samples'/('phase-'+str(index)+'-after'));sample['phase_index']=index;sample['boundary']='after';write(out/'memory-samples'/('phase-'+str(index)+'-after')/'sample.json',sample);result['owned_host_memory_samples'].append(sample)
   result['phase_receipts'].append(row);write(out/'report.progress.json',result)
   if phase['name']=='history_pinned':
    from full_cache_shared_metadata_v2 import records as read_metadata
    phase_text,_=phase_trace((out/'engine.combined.log').read_text(),phase['name'],index);rids=row['work'][0]['rid'];items=[r for r in read_metadata(phase_text,row['producer_ack']['engine_pid']) if r['kind']=='checkpoint_inventory' and r['phase']=='main_body_complete' and r['rid']==rids];require(len(items)==1,'Exact own previous history checkpoint inventory required');history_prior={'inventory':items[0],'rid':rids}
   if phase['name']=='warm':
    (out/'ARM.request').write_text('Owned actual warm terminals checked\n',encoding='ascii');limit=time.monotonic()+30
    while not (out/'ARM').exists():require(time.monotonic()<limit,'Actual same-process ARM publication failed');time.sleep(.05)
  from full_cache_shared_reply_decode_v4 import execute as decode_all_replies
  result['own_reply_decode']=decode_all_replies(plan,out,out/'own-reply-decode',result['phase_receipts'],events(out/'api-native-trace.jsonl'))
  # Real serial reference jobs are handed to a separately owned fresh arm;
  # observations are never substituted as results of that unexecuted arm.
  jobs=[j for r in result['phase_receipts'] if r['raw49'] for j in r['raw49']['jobs']];write(out/'fresh-control-jobs.json',{'jobs':jobs,'actual_fresh_execution_observed':False,'all49_comparison_required':True})
  with urllib.request.urlopen(url+'/slots/0?action=save',data=json.dumps({'filename':'parallel-negative.bin'}).encode(),timeout=30) as f:raise ValueError('Parallel persisted request unexpectedly accepted')
 except urllib.error.HTTPError as exc:
  if exc.code==501 and len(result['phase_receipts'])==len(plan['schedule']):
   from full_cache_shared_persisted_v2 import fingerprint_refusal
   try:
    body=exc.read().decode();binding=fingerprint_refusal(exc.code,json.loads(body),True);write(out/'parallel-persisted-negative.json',{'status':501,'body':body,'request_method':'POST','request_url':url+'/slots/0?action=save','request_body':{'filename':'parallel-negative.bin'},'source_refusal_binding':binding,'real_endpoint_request':True})
   except BaseException as failed:result['error']='Actual parallel persisted refusal: '+str(failed)
  else:result['error']='HTTPError: '+str(exc)
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc);result['traceback']=traceback.format_exc()
 finally:
  if started:
   try:
    from full_cache_shared_actor_retirement_v4 import retain_until_settled
    name='b70-prefix-'+str(os.getpid())+'-fullcache-shared40-v3';retirement,ledger=retain_until_settled(name,lambda:owned(plan,out,os.getpid()),subprocess.run,c1.absent,time.sleep,write,out);state=retirement['state'];removed=retirement['removed'];result['actor_retirement']=ledger;require(retirement['normal_terminal'] and ledger['failure_count']==0 and ledger['interruption_signals']==[],'Original owned actor cleanup/recovery did not qualify normal terminal')
   except BaseException as exc:result['error']=result['error'] or 'Owned teardown: '+str(exc)
  try:
   closed=[e for e in events(out/'api-native-trace.jsonl') if e['kind']=='engine_close'];require(closed and all(e['exit_code']==0 and e['error'] is None for e in closed),'Actual native QUIT/drain/exit proof missing');status=read(out/'api-native-trace.jsonl.buffered-status.json');require(status['passed'] is True and status['fullcache_phase_thread_retired'] is True and status['fullcache_control_thread_retired'] is True,'Actual owned producer phase/control/buffer EOF retirement missing');require(status['fullcache_phase_errors']==[] and status['fullcache_control_errors']==[],'Actual phase/control producer errors');result['buffered_status']=status;result['logical_owner_proofs']=__import__('full_cache_shared_source40_ownership_v4').ownership((out/'engine.combined.log').read_text(),[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],2)
   from full_cache_shared_metadata_v2 import records as metadata_records
   from full_cache_shared_memory_v2 import account
   pids={e['engine_pid'] for e in events(out/'api-native-trace.jsonl') if e['kind']=='engine_begin'};require(len(pids)==1,'Resource accounting must remain in actual single actor incarnation');from full_cache_shared_memory_capture_v4 import recollect as memory_recollect
   for sample in result['owned_host_memory_samples']:memory_recollect(out/'memory-samples'/('phase-'+str(sample['phase_index'])+'-'+sample['boundary']),sample)
   result['resource_accounting']=account(metadata_records((out/'engine.combined.log').read_text(),next(iter(pids))),result['owned_host_memory_samples']);write(out/'resource-accounting.json',result['resource_accounting'])
  except BaseException as exc:result['error']=result['error'] or 'Postterminal producer/owners: '+str(exc)
  bindings={};total=0
  for file in sorted(out.rglob('*')):
   require(not file.is_symlink(),'Shared artifact symlink')
   if not file.is_file() or file.name=='report.json':continue
   rel=file.relative_to(out).as_posix();size=file.stat().st_size
   if 'captures' in file.relative_to(out).parts:total+=size
   require(total<=plan['capture_disk_quota_bytes'],'Declared whole-suite raw disk quota exceeded');bindings[rel]={'sha256':sha(file),'bytes':size}
  result['artifact_bindings']=bindings
  result.update(state=state,removed=removed,finished_epoch=time.time(),collection_and_teardown_passed=result['error'] is None and removed and len(result['phase_receipts'])==len(plan['schedule']),producer_pid=os.getpid(),producer_interpreter=str(Path(sys.executable).resolve()),producer_interpreter_sha256=sha(sys.executable),plan_sha256=sha(out/'plan.snapshot.json'),scope=full_scope_status({}),all49_fresh_math_qualified=False,physical_memory_or_expert_residency_qualified=False,exact_checkpoint_victim_observed=False,full_model_math_qualified=False,latency_qualified=False);write(out/'report.json',result)
 for number,handler in previous_signals.items():signal.signal(number,handler)
 return result
