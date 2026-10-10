"""Root-owned concrete shared-cache HTTP phase execution; no full-cache PASS."""
import json,os,re,subprocess,sys,threading,time,traceback,urllib.request,urllib.error
from pathlib import Path
import c1_serve_controller_combined_v137 as c1
from full_cache_shared_http_client_v1 import cohort
from full_cache_shared_phase_contract_v1 import calls_and_work,phase_events,full_scope_status
from full_cache_shared_raw49_v1 import collect
from api_owned_terminal_association_v2 import associate_events
from batch_api_prefixes_v2 import prefix_jobs
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
 out=Path(out).resolve();root=Path(__file__).resolve().parents[2];here=Path(__file__).resolve().parent;prepared=read(Path(plan['prepared'])/'prepared.json');engine=Path(prepared['engine_receipt']).parent;lock=read(here/'model-lock.json');name='b70-fullcache-shared-v1-'+str(pid)
 command=['docker','run','-d','--name',name,'--label','b70.fullcache.shared.plan='+sha(out/'plan.snapshot.json'),'--label','b70.prefix.plan='+sha(out/'plan.snapshot.json'),'--entrypoint','/bin/bash','--network','host','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--group-add',str(os.stat('/dev/dri/card0').st_gid)]
 for src,dst,ro in [(engine/'build','/build',True),(engine/'source','/src',True),(Path(plan['pack']),'/pack',True),(root/lock['destination'],'/model',True),(out,'/results',False),(here,'/controller',True)]:command+=['-v',str(src)+':'+dst+(':ro' if ro else ':rw')]
 env={'B70_BATCH_TRACE':'/results/api-native-trace.jsonl','B70_BATCH_NATIVE_LOG':'/results/engine.combined.log','B70_BATCH_ARM_REQUEST':'/results/ARM.request','B70_BATCH_ARM':'/results/ARM','B70_FULLCACHE_PHASE_REQUEST':'/results/phase.request.json','B70_FULLCACHE_PHASE_ACK':'/results/phase.ack.json','STRATA_ARTIFACT_IDENTITY_SHA256':sha(out/'artifact-identity.json')}
 for k,v in sorted(env.items()):command+=['-e',k+'='+v]
 return command+[plan['image'],'-lc','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /opt/b70-c1-python/bin/python /controller/full_cache_shared_api_trace_v1.py --engine strata --config /results/server-config.json --host 127.0.0.1 --port '+str(plan['port'])]
def owned(plan,out,pid):
 name='b70-fullcache-shared-v1-'+str(pid);obj=c1.inspected(name);require(obj['Name']=='/'+name and obj['Config']['Image']==plan['image'] and obj['Config']['Labels'].get('b70.fullcache.shared.plan')==sha(Path(out)/'plan.snapshot.json'),'Actual foreign full-cache actor owner/image/label');return obj
def publish_phase(out,index,name,deadline=30):
 out=Path(out);path=out/'phase.request.json';temporary=out/'phase.request.tmp';write(temporary,{'index':index,'name':name});temporary.replace(path);limit=time.monotonic()+deadline
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
   sends=[e for e in seen if e['kind']=='native_send' and e['line'].startswith('BGEN ')];progress=[e for e in seen if e['kind']=='native_receive' and e['line'].startswith('PP ')]
   if not progress:continue
   require(len(begins)==len(sends)==1 and begins[0]['engine_pid']==ack['engine_pid']==sends[0]['engine_pid'],'Prefill PP must have sole actual owned pending admission');p=progress[0];require(p['engine_pid']==ack['engine_pid'] and sends[0]['sequence']<p['sequence'] and not any(e['kind']=='native_receive' and e['line'].startswith(('T ','BT ','DONE ','BADM ','BDONE ')) for e in seen),'Actual prefill cancellation gate reached after output/terminal');values=p['line'].split();require(0<int(values[1])<len(phase['rows'][0]['ids']),'Actual PP must precede whole prompt completion');receipt.update(kind='actual_owned_PP_before_first_token',source_record=p,sole_native_command=sends[0],engine_begin=begins[0]);write(gate,receipt);return
  body=[e for e in seen if e['kind']=='native_receive' and e['line'].startswith('SBF batch_event ') and dict(re.findall(r'(\w+)=([^ ]+)',e['line'])).get('rows')=='2']
  if body:receipt.update(kind='actual_completed_two_row_decode',source_record=body[0]);write(gate,receipt);return
def run_phase(plan,phase,out,index,url):
 directory=Path(out)/'phases'/phase['name'];directory.mkdir(parents=True,exist_ok=False);ack=publish_phase(out,index,phase['name']);write(directory/'phase.recipe.json',phase);write(directory/'producer-ack.json',ack);gate=directory/'cancel.allowed.json';stop=threading.Event();gate_receipt={};thread=None
 if phase['cancel_kind']:thread=threading.Thread(target=cancellation_watch,args=(out,phase,ack,gate,stop,gate_receipt),daemon=True);thread.start()
 try:client=cohort(url,'hotschmoe-dd',[r['messages'] for r in phase['rows']],directory/'client',cancel_index=phase['cancel_index'],max_new=phase['max_new'],cancel_gate=gate if phase['cancel_kind'] else None,abort_event=ABORT,request_policy=phase['policy'])
 finally:stop.set();thread and thread.join(timeout=5)
 require(client['client_transport_completed'],'Actual shared phase HTTP transport incomplete');limit=time.monotonic()+60
 while True:
  all_events=events(Path(out)/'api-native-trace.jsonl');current=[e for e in all_events if e['sequence']>ack['sequence']];calls={e['call'] for e in current if e['kind']=='engine_begin'}
  try:terminals=associate_events(current,calls);require(len(terminals)==len(phase['rows']),'Actual phase native/API end roster missing');break
  except (KeyError,ValueError):require(time.monotonic()<limit,'Actual producer terminals/drain failed');time.sleep(.05)
 last=max(e['sequence'] for e in current if e['kind']=='engine_end');current=phase_events(all_events,ack['sequence'],last);work=calls_and_work(phase,current);trace,bounds=phase_trace((Path(out)/'engine.combined.log').read_text(),phase['name'],index)
 if phase['cancel_kind']:
  require(gate.exists() and thread is not None and not thread.is_alive() and gate_receipt==read(gate),'Real native observation cancellation gate missing');require(sum(e['actual_client_cancelled'] for e in terminals.values())==1,'Actual single client/native cancellation required')
  if phase['cancel_kind']=='prefill':require(next(iter(terminals.values()))['actual_generated_ids']==[],'Prefill cancellation produced a token; decode cancel cannot substitute')
 if phase['native_multirow_required']:require(any(line.startswith('SBF batch_event ') and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rows')=='2' for line in trace.splitlines()),'Actual phase lacks genuine two-row body')
 if phase['name']=='prime0':require(any(line.startswith('BCPUBLIC pin_saved ') and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('rid')==str(work[0]['rid']) and dict(re.findall(r'(\w+)=([^ ]+)',line)).get('tokens')=='272' for line in trace.splitlines()),'Actual cold prime failed to save exact public boundary272')
 required={r['rid']:['admission']+(['later'] if phase['native_multirow_required'] else []) for r in work};raw=None
 if phase['name'] not in ('warm','prefill_cancel0'):raw=collect(trace,current,work,[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],Path(out)/'captures',required)
 record={'phase':phase,'producer_ack':ack,'end_sequence':last,'original_line_bounds':bounds,'work':work,'terminal_associations':terminals,'client':client,'raw49':raw,'cancel_gate':gate_receipt or None,'raw49_unobserved_reason':'unarmed warm or native partial-prefill cancellation has no completed final49' if raw is None else None,'fresh49_comparison_completed':False,'full_cache_runtime_qualified':False};write(directory/'receipt.json',record);return record
def run(plan,out,pre_health):
 import full_cache_shared_runtime_v1 as ctrl
 ctrl.manifest_binding(plan);c1.leased([0,1]);health=read(pre_health);require(health['passed'] is True and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh owned parent pairhealth required');out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir();write(out/'plan.snapshot.json',plan);prepared=Path(plan['prepared']);cfg=read(prepared/'server-config.json');manifest=read(prepared/'artifact-identity.json');cfg.update(args=plan['args'],env=plan['env'],parallel=2,port=plan['port'],model_name='hotschmoe-dd',aliases=[plan['research_alias']],log='/results/server-engine.log',slot_save_path='/results/sessions');manifest.update(primary_model_name='hotschmoe-dd',research_alias=plan['research_alias']);manifest['runtime'].update(args=plan['args'],env=plan['env']);write(out/'server-config.json',cfg);write(out/'artifact-identity.json',manifest);command=command_recipe(plan,out,os.getpid());write(out/'launch.command.json',command);result={'passed':False,'phase_receipts':[],'error':None,'started_epoch':time.time(),'full_cache_runtime_qualified':False};started=False;removed=False;state=None
 try:
  subprocess.run(command,check=True,capture_output=True,text=True,timeout=60);started=True;url='http://127.0.0.1:'+str(plan['port']);limit=time.monotonic()+900
  while True:
   state=owned(plan,out,os.getpid())['State'];require(state['Running'],'Owned fullcache actor exited before readiness')
   try:
    with urllib.request.urlopen(url+'/v1/models',timeout=5) as f:models=json.load(f)
    require([r['id'] for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==sha(out/'artifact-identity.json') for r in models['data']),'Actual live fullcache served identity differs');write(out/'models.json',models);break
   except (OSError,TimeoutError):require(time.monotonic()<limit,'Fullcache readiness deadline');time.sleep(1)
  for index,phase in enumerate(plan['schedule'],1):
   row=run_phase(plan,phase,out,index,url);result['phase_receipts'].append(row);write(out/'report.progress.json',result)
   if phase['name']=='warm':
    (out/'ARM.request').write_text('Owned actual warm terminals checked\n',encoding='ascii');limit=time.monotonic()+30
    while not (out/'ARM').exists():require(time.monotonic()<limit,'Actual same-process ARM publication failed');time.sleep(.05)
  # Real serial reference jobs are handed to a separately owned fresh arm;
  # observations are never substituted as results of that unexecuted arm.
  jobs=[j for r in result['phase_receipts'] if r['raw49'] for j in r['raw49']['jobs']];write(out/'fresh-control-jobs.json',{'jobs':jobs,'actual_fresh_execution_observed':False,'all49_comparison_required':True})
  with urllib.request.urlopen(url+'/slots/0?action=save',data=json.dumps({'filename':'parallel-negative.bin'}).encode(),timeout=30) as f:raise ValueError('Parallel persisted request unexpectedly accepted')
 except urllib.error.HTTPError as exc:
  if exc.code==501 and len(result['phase_receipts'])==len(plan['schedule']):write(out/'parallel-persisted-negative.json',{'status':501,'body':exc.read().decode(),'real_endpoint_request':True})
  else:result['error']='HTTPError: '+str(exc)
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc);result['traceback']=traceback.format_exc()
 finally:
  if started:
   try:
    owned(plan,out,os.getpid());subprocess.run(['docker','kill','--signal','TERM','b70-fullcache-shared-v1-'+str(os.getpid())],check=True,capture_output=True,timeout=30);limit=time.monotonic()+180
    while owned(plan,out,os.getpid())['State']['Running']:require(time.monotonic()<limit,'Owned normal teardown deadline');time.sleep(1)
    obj=owned(plan,out,os.getpid());state=obj['State'];write(out/'owned-terminal-inspection.json',obj);require(state['ExitCode']==0 and not state['OOMKilled'] and not state['Error'],'Fullcache owned actor abnormal terminal');subprocess.run(['docker','rm','b70-fullcache-shared-v1-'+str(os.getpid())],check=True,capture_output=True,timeout=30);removed=c1.absent('b70-fullcache-shared-v1-'+str(os.getpid()))
   except BaseException as exc:result['error']=result['error'] or 'Owned teardown: '+str(exc)
  try:
   closed=[e for e in events(out/'api-native-trace.jsonl') if e['kind']=='engine_close'];require(closed and all(e['exit_code']==0 and e['error'] is None for e in closed),'Actual native QUIT/drain/exit proof missing');status=read(out/'api-native-trace.jsonl.buffered-status.json');require(status['passed'] is True and status['fullcache_phase_thread_retired'] is True,'Actual owned producer phase/buffer EOF retirement missing');require(status['fullcache_phase_errors']==[],'Actual phase producer errors');result['buffered_status']=status;result['logical_owner_proofs']=owner_proofs((out/'engine.combined.log').read_text(),[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],2,'source37',snapshots_observed=True)
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
 return result
