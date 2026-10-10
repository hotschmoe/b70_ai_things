#!/usr/bin/env python3
"""Owned real API warm/ARM collection. Parent supplies lease, health and source watch.
This collects evidence; source identity and serial comparisons finalize separately.
"""
import argparse,json,os,re,shlex,subprocess,sys,threading,time,traceback,urllib.request
from pathlib import Path
from batch_api_client_cache_positive_v2 import cohort
from api_cache_positive_contract_v2 import profile,alias as new_alias,registry_gate,PROFILE_ENV,actual_startup,CHECKPOINT_ARGS,lock_checkpoint_defaults,last_live_handoff,actual_phase_legs
from batch_numerical_protocol_v2 import Roster
from batch_api_prefixes_v2 import prefix_jobs
from batch_numerical_proofs_v7 import genuine_baseline,providers,require,read,write,sha,raw_case,terminal_api_join,owner_proofs,artifact_bindings,source_observers_off
ABORT=threading.Event()
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def set_arg(args,key,value):
 if key in args:args[args.index(key)+1]=str(value)
 else:args.extend([key,str(value)])

def experimental_alias(cfg,slots,diagnostic,lane='source35'):
 require(lane=='source35','Exact source35 required');return new_alias(cfg,slots,diagnostic)

def derived_config(prepared_dir,out,slots,port,diagnostic,lane='source35'):
 """Close actual derived args/env under the same exact source-model manifest."""
 cfg=read(prepared_dir/'server-config.json');manifest=read(prepared_dir/'artifact-identity.json');args=list(cfg['args']);env=dict(cfg['env']);lock_checkpoint_defaults(args,env)
 source_observers_off(env);env.update(STRATA_PLE_INPUT33='0',STRATA_PREFIX30='0')
 for key,val in [('--max-context',2048),('--prefill',64),('--batch',slots),('--batch-groups',1),('--prompt-cache',3),('--conversation-cache-mib',0),('--adapt-every',0)]:set_arg(args,key,val)
 require('--mtp' not in args and '--pipeline-windows' not in args,'No MTP/pipeline activated pilot')
 for key,val in CHECKPOINT_ARGS.items():set_arg(args,key,val)
 for key in ['STRATA_FIDELITY_DIAG','STRATA_LAYER0_Q8_DIAG']:env[key]='0'
 require('STRATA_CKPT_REREAD' not in env,'Checkpoint reread env presence invalidates exact -1 policy')
 env['STRATA_MIRROR_OWNER_TRACE']='1' if lane=='source35' else '0'
 env.update(PROFILE_ENV);env.update(STRATA_SLOT_OWNER_TRACE='1',STRATA_BATCH_FIDELITY_DIAG=str(int(diagnostic)),STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',SYCL_UR_TRACE='2')
 cfg.update(args=args,env=env,port=port,parallel=slots,log='/results/server-engine.log');alias=experimental_alias(cfg,slots,diagnostic,lane);cfg['model_name']='hotschmoe-dd';cfg['aliases']=[alias];manifest['primary_model_name']='hotschmoe-dd';manifest['research_alias']=alias
 manifest['runtime']['args']=args;manifest['runtime']['env']={k:str(v) for k,v in sorted(env.items()) if k.startswith(('STRATA_','SYCL_','ONEAPI_')) and k!='STRATA_ARTIFACT_IDENTITY_SHA256'}
 write(out/'artifact-identity.json',manifest);write(out/'server-config.json',cfg);return cfg,manifest

def jsonlines(path):
 if not path.is_file():return []
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def command_recipe(plan,out,pid):
 out=Path(out);prepared=read(Path(plan['prepared'])/'prepared.json');slots=plan['slots'];diagnostic=bool(plan['diagnostic']);container='b70-prefix-'+str(pid)+'-api'+str(slots)+('-on' if diagnostic else '-off');identity=sha(out/'artifact-identity.json')
 env={'B70_BATCH_TRACE':'/results/api-native-trace.jsonl','B70_BATCH_NATIVE_LOG':'/results/engine.combined.log','B70_BATCH_ARM_REQUEST':'/results/ARM.request','B70_BATCH_ARM':'/results/ARM','STRATA_ARTIFACT_IDENTITY_SHA256':identity}
 engine=Path(prepared['engine_receipt']).parent;lock=read(HERE/'model-lock.json')
 command=['docker','run','-d','--name',container,'--label','b70.prefix.plan='+sha(out/'plan.snapshot.json'),'--network','host','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--group-add',str(os.stat('/dev/dri/card0').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',prepared['pack']+':/pack:ro','-v',str(ROOT/lock['destination'])+':/model:ro','-v',str(out.resolve())+':/results','-v',str(HERE)+':/controller:ro']
 for k,v in sorted(env.items()):command+=['-e',k+'='+v]
 command += [prepared['runtime']['image'],'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /opt/b70-c1-python/bin/python /controller/batch_api_trace_v2.py --engine strata --config /results/server-config.json --host 127.0.0.1 --port '+str(plan['port'])]
 return command


def seal_report(plan,out,result):
 require(read(out/'plan.snapshot.json')==plan,'Actual child snapshot content differs before sealing')
 result.update(producer_interpreter=str(Path(sys.executable).resolve()),producer_interpreter_sha256=sha(Path(sys.executable).resolve()),producer_pid=os.getpid())
 result.update(plan_sha256=sha(out/'plan.snapshot.json'),schema=6,harness_generation=9,api_cache_positive_generation=2,actual_cached_state_handoff_qualified=False);result['artifact_bindings']=artifact_bindings(out);write(out/'report.json',result)
 return result


def run(plan,out,pre_health,diagnostic=True):
 source_observers_off(plan['env']);profile(plan)
 c1,parent=providers(plan['lane']);c1.leased([0,1]);prepared,binding=genuine_baseline(Path(plan['prepared']),plan['lane']);require(binding['prepared_sha256']==plan['prepared_sha256'],'Prepared baseline changed')
 slots=plan['slots'];require(slots in (2,4,6),'Actual requested capacity must2/4/6');health=read(pre_health);require(health['passed'] and set(prepared['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Actual fresh parent prehealth required')
 out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir();write(out/'plan.snapshot.json',plan);cfg,manifest=derived_config(Path(plan['prepared']),out,slots,plan['port'],diagnostic,plan['lane']);registration=registry_gate(cfg['aliases'][0]);require(registration==plan['registry_binding'],'Derived registry identity changed');container='b70-prefix-'+str(os.getpid())+'-api'+str(slots)+('-on' if diagnostic else '-off');identity=sha(out/'artifact-identity.json')
 command=command_recipe(plan,out,os.getpid())
 write(out/'launch.command.json',command);error=None;result={};started=False;state=None;removed=False;cancel_gate=out/'CANCEL.allowed';stop_watch=threading.Event()
 try:
  subprocess.run(command,capture_output=True,text=True,check=True,timeout=60);started=True;url='http://127.0.0.1:'+str(plan['port']);deadline=time.monotonic()+900
  while True:
   state=c1.inspected(container)['State'];require(state['Running'],'API process exited before readiness')
   try:
    with urllib.request.urlopen(url+'/v1/models',timeout=5) as f:models=json.load(f)
    require([row['id'] for row in models['data']]==['hotschmoe-dd',cfg['aliases'][0]] and all(row['meta']['artifact_identity']['artifact_identity_sha256']==identity for row in models['data']),'Live API identity differs')
    break
   except (OSError,TimeoutError):
    require(time.monotonic()<deadline,'API ready deadline exceeded');time.sleep(1)
  write(out/'models.json',models)
  warm=cohort(url,'hotschmoe-dd',plan['messages']['warm'],out/'warm',max_new=32,abort_event=ABORT,request_policy=plan['API_warm_request_policy']);require(warm['client_transport_completed'],'Actual warm API transport incomplete')
  trace_path=out/'engine.combined.log';warm_events=jsonlines(out/'api-native-trace.jsonl');warm_begins={r['call']:r for r in warm_events if r['kind']=='engine_begin'};warm_ends={r['call']:r for r in warm_events if r['kind']=='engine_end'}
  require(len(warm_begins)==slots and set(warm_begins)==set(warm_ends) and all(not e['cancelled'] and e['error'] is None for e in warm_ends.values()),'Warm API/native terminals incomplete')
  capacity=Roster(slots)
  for e in warm_events:
   if e['kind']=='native_receive' and e['line'].startswith('INFO '):capacity.consume(e['line'])
  result['actual_native_startup']=actual_startup(trace_path.read_text(),slots);result['actual_warm_phase_legs']=actual_phase_legs(warm_events,set(warm_begins),True);capacity.capacity();require(capacity.info.get('slot_cache')==1,'Actual API solo migration requires native slot state cache capability')
  require({json.dumps(e['rendered_prompt']['messages'],sort_keys=True):e['submitted_ids'] for e in warm_begins.values()}=={json.dumps(messages,sort_keys=True):ids for messages,ids in zip(plan['messages']['warm'],plan['api_token_ids']['warm'])},'Actual warm API tokenizer/template IDs differ from pinned CPU spec')
  warm_prefixes=prefix_jobs(warm_events);require(warm_prefixes['all_segments_terminal'],'Warm native segment still live')
  if diagnostic:
   require(warm_prefixes['actual_completed_multirow_events']>0,'Warm API did not execute actual unarmed multirow body');require(not any(line.startswith(('SBF request ','SBF vector ','SBF replay ','SBF allocation ')) for line in trace_path.read_text().splitlines()),'Unarmed API observer operations occurred')
   (out/'ARM.request').write_text('Actual warm client and native terminals checked\n',encoding='ascii');deadline=time.monotonic()+30
   while not (out/'ARM').is_file():require(time.monotonic()<deadline,'Actor ARM marker deadline');time.sleep(.05)
  if not diagnostic:
   (out/'ARM.request').write_text('Off-run target boundary after native warm terminals\n',encoding='ascii');deadline=time.monotonic()+30
   while not (out/'ARM').is_file():require(time.monotonic()<deadline,'Off actor target marker deadline');time.sleep(.05)
  def allow_cancel():
   while not stop_watch.is_set():
    trace=trace_path.read_text() if trace_path.exists() else '';tail=trace.split('HARNESS ARM')[-1]
    matched=[dict(re.findall(r'(\w+)=([^ ]+)',l)) for l in tail.splitlines() if l.startswith('SBF batch_event ')]
    if diagnostic and any(int(e['rows'])==slots and e['completed']=='1' for e in matched):cancel_gate.write_text('Actual requested-N body completed\n');return
    if not diagnostic:
     # No observer is enabled. Cancellation is gated by real BT traffic from
     # at least two private slots; N-row proof belongs only to the on run.
     bt={line.split()[1] for line in tail.splitlines() if line.startswith('BT ')}
     if len(bt)>=2:cancel_gate.write_text('Real private-slot traffic observed\n');return
    stop_watch.wait(.05)
  watcher=threading.Thread(target=allow_cancel,daemon=True);watcher.start();target=cohort(url,'hotschmoe-dd',plan['messages']['target'],out/'target',plan['cancel_index'],plan['max_new_by_request'],cancel_gate=cancel_gate,abort_event=ABORT,request_policy=plan['API_target_request_policy']);stop_watch.set();watcher.join(timeout=5);require(target['client_transport_completed'] and target['real_client_cancel_requested'],'Actual target API transport/cancel incomplete')
  events=jsonlines(out/'api-native-trace.jsonl');begins={e['call']:e for e in events if e['kind']=='engine_begin' and e['call'] not in warm_begins};ends={e['call']:e for e in events if e['kind']=='engine_end' and e['call'] in begins};require(len(begins)==slots and set(begins)==set(ends),'Target request-local completion roster missing')
  cancelled={call for call,e in ends.items() if e['cancelled']};require(len(cancelled)==1,'Exactly real canceled target required');api=terminal_api_join([e for e in events if e.get('call') in begins or e['kind']=='native_receive'],set(begins),cancelled)
  jobs=prefix_jobs(events);target_jobs=[j for j in jobs['jobs'] if j['call'] in begins];write(out/'serial-jobs.json',{'jobs':target_jobs});write(out/'prefixes.json',jobs)
  logical={}
  for index,messages in enumerate(plan['messages']['target']):
   matched=[call for call,e in begins.items() if e['rendered_prompt']['messages']==messages];require(len(matched)==1,'Logical target messages not uniquely bound to actual API call');require(begins[matched[0]]['submitted_ids']==plan['api_token_ids']['target'][index],'Actual target API tokenizer/template IDs differ from pinned CPU spec');logical[index]=ends[matched[0]]['rid']
  if diagnostic:
   rids=[ends[call]['rid'] for call in begins];migrating=sorted({j['rid'] for j in target_jobs if j['role']=='solo_migration'});require(migrating,'No actual surviving request completed native solo migration')
   stages=[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])]
   policy={}
   for row in plan['actual_counter_policy']:
    rid=logical[row['logical_index']];job=[j for j in target_jobs if j['rid']==rid and j['role']==row['role']];require(len(job)==1,'Preregistered counter role lacks exact actual consumed-prefix job');policy[rid,row['role']]={key:len(job[0]['ids'])-1 if value=='prompt_minus_one' else value for key,value in row['values'].items()}
   result['raw']=raw_case(trace_path.read_text(),events,rids,migrating,stages,out/'captures',policy,ownership_terminal=False)
  result['actual_target_cached_legs']=actual_phase_legs(events,set(begins),False);result['actual_last_live_handoffs']=[last_live_handoff(trace_path.read_text(),rid,next(j for j in target_jobs if j['rid']==rid and j['role']=='solo_migration'),len(plan['stage_ranges'])) for rid in migrating] if diagnostic else [];result.update(api=api,client=target,initial_native_and_api_collection_complete=True,diagnostic=diagnostic)
 except BaseException as exc:error=type(exc).__name__+': '+str(exc);failure_traceback=traceback.format_exc()
 finally:
  stop_watch.set()
  if started:
   subprocess.run(['docker','kill','--signal','TERM',container],capture_output=True,text=True,timeout=30);deadline=time.monotonic()+180
   while c1.inspected(container)['State']['Running'] and time.monotonic()<deadline:time.sleep(1)
   state=c1.inspected(container)['State']
   with (out/'container.log').open('w') as log:subprocess.run(['docker','logs',container],stdout=log,stderr=subprocess.STDOUT,timeout=30)
   if not state['Running']:subprocess.run(['docker','rm',container],capture_output=True,text=True,check=True,timeout=30);removed=c1.absent(container)
   closed=[e for e in jsonlines(out/'api-native-trace.jsonl') if e['kind']=='engine_close'];require_close=bool(closed) and all(e['exit_code']==0 and e['error'] is None for e in closed)
  else:require_close=False
  if error is None:
   try:result['logical_owner_proofs']=owner_proofs((out/'engine.combined.log').read_text(),[(i,lo,hi) for i,(lo,hi) in enumerate(plan['stage_ranges'])],slots,plan['lane'],snapshots_observed=bool(diagnostic))
   except Exception as exc:error='Post-terminal ownership proof: '+str(exc)
  result.update(passed=False,collection_and_teardown_passed=error is None and require_close and removed and state['ExitCode']==0 and not state.get('OOMKilled'),error=error,actual_native_exit_zero=require_close,state=state,removed=removed,finished_epoch=time.time(),full_source_finalization_required=True,serial_raw_comparisons_completed=False,serving_mirror_owner_logical_frees=bool(result.get('logical_owner_proofs',{}).get('serving_mirror_owner_logical_frees')),full_cache_public_fresh_pin_qualified=False,full_model_math_qualified=False);seal_report(plan,out,result)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pre-health',type=Path,required=True);ap.add_argument('--diagnostic',type=int,choices=[0,1],required=True);a=ap.parse_args();r=run(read(a.plan),a.output,a.pre_health,bool(a.diagnostic));return 0 if r['collection_and_teardown_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
