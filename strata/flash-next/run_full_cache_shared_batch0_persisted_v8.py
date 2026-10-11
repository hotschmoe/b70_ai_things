"""ROOT ONLY serial HTTP/session actor; fresh49 still independently required."""
import json,os,signal,subprocess,sys,time,traceback,urllib.request,urllib.error
from pathlib import Path
import full_cache_shared_batch0_persisted_v8 as ctrl
import run_full_cache_shared_runtime_v8 as common
from full_cache_shared_http_client_v2 import cohort
from full_cache_shared_batch0_pcl_v8 import full49
from full_cache_shared_persisted_lineage_v8 import negative_file,session_binding,saved_files,refusal_state_unchanged


def command_recipe(plan,out,pid):
 command=common.command_recipe(plan,out,pid);needle='/controller/full_cache_shared_api_trace_v8.py';ctrl.require(command[-1].count(needle)==1,'Exact source40 serial observer command boundary required');command[-1]=command[-1].replace(needle,'/controller/full_cache_shared_batch0_api_trace_v8.py');return command

def phase_stream(plan,out,ack):
 from full_cache_semantic_phase_stream_v1 import PhaseStream
 from full_cache_shared_memory_capture_v8 import recipe_binding
 command=command_recipe(plan,out,os.getpid());name=command[command.index('--name')+1];obj=ctrl.c1.inspected(name);ctrl.require(obj['Name']=='/'+name and obj['Config']['Labels'].get('b70.prefix.plan')==ctrl.sha(Path(out)/'plan.snapshot.json'),'Exact serial phase cursor owner differs');recipe_binding(obj,command,plan['image']);return PhaseStream(out,ack,obj)
def request(plan,name,out,index,url):
 spec=plan['schedule'][name];row=spec['row'];directory=Path(out)/'phases'/name;directory.mkdir(parents=True);ack=common.publish_phase(out,index,name);stream=phase_stream(plan,out,ack);client=cohort(url,'hotschmoe-dd',[row['messages']],directory/'client',max_new=[1],abort_event=common.ABORT,request_policy={'strata_fresh':bool(spec['fresh'])});ctrl.require(client['client_transport_completed'],'Original serial HTTP request transport failed');limit=time.monotonic()+60
 while True:
  events=[e for e in stream.poll() if e['sequence']>ack['sequence']];begins=[e for e in events if e['kind']=='engine_begin'];ends=[e for e in events if e['kind']=='engine_end']
  if len(begins)==len(ends)==1:
   try:
    expected={k:spec[k] for k in ('fresh','pin','expected_reused')};expected['ids']=row['ids'];numeric=full49(events,begins[0],ends[0],expected,Path(out)/'captures',ctrl.expected_stage_ranges(plan['args']));break
   except (KeyError,ValueError)as error:ctrl.write(directory/'last-terminal-predicate-error.json',{'error':type(error).__name__+': '+str(error),'unchanged_terminal_drain_seconds':60});ctrl.require(time.monotonic()<limit,'Actual complete serial PCL/full49 coverage failed: '+str(error));time.sleep(.05)
  else:ctrl.require(time.monotonic()<limit,'Actual FIFO serial begin/end missing');time.sleep(.05)
 from full_cache_shared_batch0_decode_v8 import execute as own_decode
 decode=own_decode(plan,Path(out),Path(out)/'serial-decodes'/name,row['messages'],client['rows'][0],begins[0],ends[0],numeric['source_lineage'])
 record={'phase':name,'spec':spec,'producer_ack':ack,'end_sequence':max(e['sequence'] for e in events),'client':client,'native_begin':begins[0],'native_end':ends[0],'numeric':numeric,'independent_ID_text_decode':decode,'actual_independent_fresh49_still_required':True};ctrl.write(directory/'semantic-stream.json',stream.snapshot());ctrl.write(directory/'receipt.json',record);stream.close();return record

def session(plan,name,out,index,url,owner):
 spec=plan['schedule'][name];directory=Path(out)/'phases'/name;directory.mkdir(parents=True);ack=common.publish_phase(out,index,name);stream=phase_stream(plan,out,ack);action='save' if spec['action']=='save' else 'restore';body={'filename':spec['filename']};req=urllib.request.Request(url+'/slots/0?action='+action,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST');started=time.time()
 try:
  with urllib.request.urlopen(req,timeout=3600) as response:status=response.status;raw=response.read()
 except urllib.error.HTTPError as error:status=error.code;raw=error.read()
 http={'request':{'method':'POST','path':'/slots/0?action='+action,'body':body},'status':status,'body':json.loads(raw),'raw_body_sha256':__import__('hashlib').sha256(raw).hexdigest(),'error':None,'started_epoch':started,'finished_epoch':time.time()};(directory/'HTTP.body.json').write_bytes(raw);ctrl.write(directory/'HTTP.receipt.json',http);events=[e for e in stream.poll() if e['sequence']>ack['sequence']];binding=session_binding(events,http,spec,owner);record={'phase':name,'spec':spec,'producer_ack':ack,'end_sequence':max(e['sequence'] for e in events),'HTTP':http,'native_HTTP_binding':binding};ctrl.write(directory/'semantic-stream.json',stream.snapshot());ctrl.write(directory/'receipt.json',record);stream.close();return record

def run(plan,out,health_path,plan_bytes,handoff,seal):
 ctrl.source_binding();ctrl.c1.leased([0,1]);health=ctrl.read(health_path);ctrl.require(health==handoff['actual_health'] and health['passed'] is True and 0<=time.time()-health['finished_epoch']<=300,'Actual fresh postsemantic health required');out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir();(out/'sessions').mkdir();(out/'ARM').write_text('Own serial PCL/full49 diagnostic\n');__import__('full_cache_shared_plan_snapshot_v8').write_runtime(out/'plan.snapshot.json',plan,plan_bytes)
 prepared=Path(plan['prepared']);cfg=ctrl.read(prepared/'server-config.json');artifact=ctrl.read(prepared/'artifact-identity.json');cfg.update(args=plan['args'],env=plan['env'],parallel=1,port=plan['port'],model_name='hotschmoe-dd',aliases=[plan['research_alias']],log='/results/server-engine.log',slot_save_path='/results/sessions');artifact.update(primary_model_name='hotschmoe-dd',research_alias=plan['research_alias']);artifact['runtime'].update(args=plan['args'],env=__import__('run_full_cache_shared_runtime_v8').runtime_identity_environment(plan['env']));ctrl.write(out/'server-config.json',cfg);ctrl.write(out/'artifact-identity.json',artifact);command=command_recipe(plan,out,os.getpid());ctrl.write(out/'launch.command.json',command)
 result={'passed':False,'health_handoff':handoff,'post_ACK_byte_seal':seal,'started_epoch':time.time(),'error':None,'request_receipts':[],'session_receipts':[],'full_cache_runtime_qualified':False};started=False;state=None;removed=False;common.ABORT.clear();previous={}
 def interrupted(number,frame):common.ABORT.set();raise KeyboardInterrupt('Owned serial actor interrupted '+str(number))
 for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous[number]=signal.signal(number,interrupted)
 def owned():
  from full_cache_shared_memory_capture_v8 import recipe_binding
  name=command[command.index('--name')+1];obj=ctrl.c1.inspected(name);ctrl.require(obj['Name']=='/'+name and obj['Config']['Labels'].get('b70.prefix.plan')==ctrl.sha(out/'plan.snapshot.json'),'Actual serial actor ownership differs');recipe_binding(obj,command,plan['image']);return obj
 try:
  from finite_tokenizer_owned_execution_v2 import execute as settle_launch
  launchroot=out/'owned-launch-client';launchroot.mkdir();started=True
  def inspect_owner(obj):
   from full_cache_shared_memory_capture_v8 import recipe_binding
   ctrl.require(obj['Name']=='/'+command[command.index('--name')+1] and obj['Config']['Labels'].get('b70.prefix.plan')==ctrl.sha(out/'plan.snapshot.json'),'Original launch owner differs');recipe_binding(obj,command,plan['image'])
  launch=settle_launch(command,launchroot,inspect_owner,timeout=60);ctrl.write(launchroot/'receipt.json',launch);result['owned_launch_client']=launch;ctrl.require(launch['return_code']==0 and launch['error'] is None and launch['stop_signals']==[],'Original detached Docker launch did not settle normally');url='http://127.0.0.1:'+str(plan['port']);limit=time.monotonic()+900
  while True:
   ctrl.require(owned()['State']['Running'],'Original serial actor died before readiness')
   try:
    with urllib.request.urlopen(url+'/v1/models',timeout=5) as response:models=json.load(response)
    ctrl.require([r['id']for r in models['data']]==['hotschmoe-dd',plan['research_alias']] and all(r['meta']['artifact_identity']['artifact_identity_sha256']==ctrl.sha(out/'artifact-identity.json') for r in models['data']),'Original serial model identity differs');ctrl.write(out/'models.json',models);break
   except (OSError,TimeoutError):ctrl.require(time.monotonic()<limit,'Owned serial model readiness deadline');time.sleep(1)
  first=request(plan,'prime',out,1,url);result['request_receipts'].append(first);owner={'engine_pid':first['native_begin']['engine_pid'],'engine_generation':first['native_begin']['engine_generation']};save=session(plan,'save',out,2,url,owner);result['session_receipts'].append(save);result['negative_derivation']=negative_file(out/'sessions/valid.bin',out/'sessions/wrong.bin');ctrl.write(out/'negative-file-derivation.json',result['negative_derivation']);wrong=session(plan,'wrong_restore',out,3,url,owner);result['session_receipts'].append(wrong);after=request(plan,'after_wrong',out,4,url);result['request_receipts'].append(after);end=next(e for e in common.events(out/'api-native-trace.jsonl') if e['kind']=='session_end' and e['sequence']<=wrong['end_sequence'] and e['path']=='/results/sessions/wrong.bin');result['refusal_unchanged49']=refusal_state_unchanged(first['numeric'],after['numeric'],end,plan['schedule']['prime']['row']['ids']);valid=session(plan,'valid_restore',out,5,url,owner);result['session_receipts'].append(valid);restored=request(plan,'after_valid',out,6,url);result['request_receipts'].append(restored)
  from full_cache_shared_raw49_v2 import compare49
  result['valid_restore49']=compare49(first['numeric'],restored['numeric']);ctrl.require(result['valid_restore49']['all49_bitwise_equal'],'Valid native restore changed original full49 response');result['saved_files_binding']=saved_files(out,result['negative_derivation'],plan['schedule']['save']['expected_saved_tokens'],first['numeric']['input_ids']);ctrl.write(out/'fresh-control-jobs.json',{'jobs':[{'phase':r['phase'],'input_ids':r['numeric']['input_ids']}for r in result['request_receipts']],'actual_independent_fresh_execution_observed':False})
 except BaseException as error:result['error']=type(error).__name__+': '+str(error);result['failure_traceback']=traceback.format_exc()
 finally:
  if started:
   try:
    from full_cache_shared_actor_retirement_v8 import retain_until_settled
    name=command[command.index('--name')+1];retirement,ledger=retain_until_settled(name,owned,subprocess.run,ctrl.c1.absent,time.sleep,ctrl.write,out);state=retirement['state'];removed=retirement['removed'];result['actor_retirement']=ledger;ctrl.require(retirement['normal_terminal'] and ledger['failure_count']==0 and ledger['interruption_signals']==[],'Original serial actor cleanup/recovery did not qualify normal terminal')
   except BaseException as error:result['error']=result['error'] or 'Owned serial teardown: '+str(error)
  try:
   all_events=common.events(out/'api-native-trace.jsonl');closed=[e for e in all_events if e['kind']=='engine_close'];ctrl.require(len(closed)==1 and closed[0]['exit_code']==0 and closed[0]['error'] is None,'Original serial native QUIT/EOF missing');status=ctrl.read(out/'api-native-trace.jsonl.buffered-status.json');ctrl.require(status['passed'] is True and status['closed'] is True and status['fullcache_phase_thread_retired'] is True and status['fullcache_control_thread_retired'] is True and status['fullcache_phase_errors']==[] and status['fullcache_control_errors']==[],'Original producer marker/control/buffer retirement failed');result['buffered_status']=status
  except BaseException as error:result['error']=result['error'] or 'Original serial source EOF: '+str(error)
  result.update(state=state,removed=removed,producer_pid=os.getpid(),producer_interpreter=str(Path(sys.executable).resolve()),producer_interpreter_sha256=ctrl.sha(sys.executable),plan_sha256=ctrl.sha(out/'plan.snapshot.json'),finished_epoch=time.time(),collection_and_teardown_passed=result['error'] is None and removed and len(result['request_receipts'])==len(result['session_receipts'])==3,independent_fresh49_qualified=False,full_cache_runtime_qualified=False,full_model_math_qualified=False,physical_memory_or_expert_residency_qualified=False)
  result['artifact_bindings']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':ctrl.sha(p)}for p in out.rglob('*') if p.is_file() and p.name!='report.json'};ctrl.write(out/'report.json',result)
  for number,handler in previous.items():signal.signal(number,handler)
 return result
