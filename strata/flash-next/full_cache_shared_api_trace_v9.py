#!/usr/bin/env python3
"""NEW buffered actual API/native observer; unchanged request/scheduler/math semantics."""
import hashlib,json,os,struct,subprocess,sys,threading,time
from buffered_native_trace_epoch_v5 import BufferedTrace
from pathlib import Path

def token_sha(ids):return hashlib.sha256(b''.join(struct.pack('<I',int(t)) for t in ids)).hexdigest()

class PipeObserver:
 def __init__(self,raw,callback,direction):self.raw=raw;self.callback=callback;self.direction=direction
 def __getattr__(self,name):return getattr(self.raw,name)
 def write(self,text):
  for line in text.splitlines():self.callback(self.direction,line)
  return self.raw.write(text)
 def __iter__(self):
  for line in self.raw:self.callback(self.direction,line.rstrip('\n'));yield line
 def readline(self,*a):
  line=self.raw.readline(*a)
  if line:self.callback(self.direction,line.rstrip('\n'))
  return line

def main():
 sys.path.insert(0,'/src');from serve import server
 from c1_trace_contract import pinned_eos
 cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());eos=pinned_eos(cfg);trace=Path(os.environ['B70_BATCH_TRACE']);combined=Path(os.environ['B70_BATCH_NATIVE_LOG']);lock=threading.Lock();local=threading.local();counter=[0];seq=[0]
 sink=BufferedTrace(trace,combined,interval=0.1,max_total_bytes=int(os.environ['B70_FULLCACHE_TEXT_BUDGET']));stop_arm=threading.Event();active_calls=set();engine_pid=[None];engine_holder=[None];service_holder=[None];phase_stop=threading.Event();phase_errors=[]
 def emit(row):return sink.emit(row,semantic=row.get('kind') in ('engine_begin','engine_end','engine_close'))
 # Parent requests ARM only after actual client/native warm terminals. The
 # producer observer owns the shared log lock, so this marker cannot split a UR
 # line or move ahead of a previously received producer line.
 arm_request=Path(os.environ['B70_BATCH_ARM_REQUEST']);arm=Path(os.environ['B70_BATCH_ARM'])
 def arm_control():
  while not arm_request.is_file():
   if stop_arm.wait(0.05):return
  sink.marker('HARNESS ARM after actual unarmed multi-row terminal')
  arm.write_text('ARM actual same-process warm terminal\n',encoding='ascii')
 arm_thread=threading.Thread(target=arm_control,name='owned-api-ARM-marker',daemon=True);arm_thread.start()
 phase_request=Path(os.environ['B70_FULLCACHE_PHASE_REQUEST']);phase_ack=Path(os.environ['B70_FULLCACHE_PHASE_ACK'])
 encode_original=server.Service.encode_prompt
 def phase_control():
  last=0
  while not phase_stop.wait(.025):
   if not phase_request.is_file():continue
   row=json.loads(phase_request.read_text());index=row['index'];name=row['name']
   if index==last:continue
   if type(index)is not int or index!=last+1 or not __import__('re').fullmatch(r'[a-z][a-z0-9_]{0,63}',name):raise ValueError('Actual fullcache phase ordering/name invalid')
   with lock:
    if active_calls:raise ValueError('Fullcache phase marker while actual API calls active')
    if set(row) not in ({'index','name'},{'index','name','render_request'}):raise ValueError('Undeclared phase control fields')
    rendered=None
    if 'render_request' in row:
     from full_cache_shared_history_v2 import render_binding,canonical_sha,digest
     request=row['render_request']
     if service_holder[0] is None or engine_pid[0] is None:raise ValueError('History render requires already observed own Service and engine')
     if type(request)is not dict or set(request)!={'messages','messages_sha256','template_kwargs'} or request['template_kwargs']!={'enable_thinking':False} or request['messages_sha256']!=canonical_sha(request['messages']):raise ValueError('History render request changed')
     # Pure original Service rendering/tokenization only. It does not dispatch
     # an engine request or use native captures as input.
     ids=list(encode_original(service_holder[0],request['messages'],None,request['template_kwargs']))
     rendered=render_binding(request,{'messages':request['messages'],'messages_sha256':request['messages_sha256'],'template_kwargs':request['template_kwargs'],'tools':None,'ids':ids,'ids_sha256':digest(ids),'engine_pid':engine_pid[0],'original_encode_prompt_called':True},engine_pid[0])
    with sink.lock:
     sink.marker('HARNESS FULLCACHE_PHASE index='+str(index)+' name='+name)
     marker=sink.emit({'kind':'fullcache_phase_begin','index':index,'name':name,'engine_pid':engine_pid[0],'render_observation':rendered},semantic=True);sequence=marker['sequence'];trace_anchor=sink.anchor(marker)
    temporary=phase_ack.with_suffix('.tmp');temporary.write_text(json.dumps({'index':index,'name':name,'sequence':sequence,'engine_pid':engine_pid[0],'render_observation':rendered,'trace_anchor':trace_anchor},ensure_ascii=True)+'\n',encoding='ascii');temporary.replace(phase_ack);last=index
 def phase_worker():
  try:phase_control()
  except BaseException as exc:phase_errors.append(type(exc).__name__+': '+str(exc))
 phase_thread=threading.Thread(target=phase_worker,name='owned-fullcache-phase-marker',daemon=True);phase_thread.start()
 control_request=Path(os.environ['B70_FULLCACHE_CONTROL_REQUEST']);control_ack=Path(os.environ['B70_FULLCACHE_CONTROL_ACK']);control_errors=[]
 def control_worker():
  last=0
  try:
   while not phase_stop.wait(.025):
    if not control_request.is_file():continue
    row=json.loads(control_request.read_text());index=row['index']
    if index==last:continue
    if set(row)!={'index','kind','proposal'} or type(index)is not int or index!=last+1 or row['kind']!='stale_BSTOP':raise ValueError('Exact preregistered stale control only')
    from full_cache_shared_stale_owner_v2 import admit_proposal
    # The original buffered trace is an observation, never an input to model
    # arithmetic. Replay submitted owner grammar before negative control send.
    observed=[json.loads(line) for line in trace.read_bytes().splitlines()];proposal=admit_proposal(row['proposal'],observed)
    with lock:
     engine=engine_holder[0]
     if not active_calls or engine is None or engine.proc.pid!=proposal['engine_pid'] or engine.gen!=proposal['current_owner']['engine_generation']:raise ValueError('Actual active same-incarnation stale control owner missing')
    local.fullcache_control=True
    try:engine._send(proposal['command'])
    finally:local.fullcache_control=False
    ack={'index':index,'kind':row['kind'],'proposal':proposal,'engine_pid':engine.proc.pid};temporary=control_ack.with_suffix('.tmp');temporary.write_text(json.dumps(ack,ensure_ascii=True)+'\n',encoding='ascii');temporary.replace(control_ack);last=index
  except BaseException as exc:control_errors.append(type(exc).__name__+': '+str(exc))
 control_thread=threading.Thread(target=control_worker,name='owned-fullcache-stale-control',daemon=True);control_thread.start()
 def encode(self,messages,tools,kwargs):
  with lock:
   if service_holder[0] is not None and service_holder[0] is not self:raise ValueError('Different Service instance cannot borrow history actor')
   service_holder[0]=self
  ids=encode_original(self,messages,tools,kwargs);local.prompt={'ids':list(ids),'sha256':token_sha(ids),'messages':messages,'tools':tools,'template_kwargs':kwargs};return ids
 popen_original=server.popen
 def popen(name,command,*a,**kw):
  if '--serve' not in command:return popen_original(name,command,*a,**kw)
  # One producer OS stream before the real API readers; preserving their routing.
  kw['stderr']=subprocess.STDOUT;proc=popen_original(name,command,*a,**kw);engine_pid[0]=proc.pid
  def native(direction,line):
   row={'kind':'fullcache_control_send' if direction=='send' and getattr(local,'fullcache_control',False) else 'native_'+direction,'engine_pid':proc.pid,'call':getattr(local,'call',None) if direction=='send' else None,'line':line}
   semantic=row['kind']=='fullcache_control_send' or line.startswith(('STOP','BSTOP ','BYIELD ','QUIT','DONE ','BDONE ','BADM ','READY','INFO ','ERR','FATAL','YIELDED ','SERR ','SESSION ','SWAIT ','SAVED ','RESTORED ','SBF slot_closed ','PCL '))
   sink.emit(row,combined_line=line if direction=='receive' else None,semantic=semantic)
  proc.stdin=PipeObserver(proc.stdin,native,'send');proc.stdout=PipeObserver(proc.stdout,native,'receive');return proc
 from eos_waiter_observer40_v1 import install as waiter_install
 waiter_observer=waiter_install(server,emit,lambda engine:{'call':getattr(local,'call',None)})
 waiter_stop=threading.Event();waiter_errors=[];waiter_request=Path('/results/waiter-proof.request.json');waiter_ack=Path('/results/waiter-proof.ack.json')
 def waiter_worker():
  previous=None
  while not waiter_stop.wait(.025):
   if not waiter_request.is_file():continue
   packet=__import__('serial37_canonical_json_v3').read_unique(waiter_request)
   if packet==previous:continue
   if set(packet)!={'index','name','calls'}or type(packet['index'])is not int or packet['index']<=0 or type(packet['calls'])is not list or not packet['calls']or any(type(c)is not int or c<=0 for c in packet['calls']):raise ValueError('Exact actual phase waiter proof request required')
   proof=waiter_observer.proof(packet['calls'])
   if not proof['all_waiter_threads_retired']:continue
   value={'request':packet,'proof':proof};event=sink.emit({'kind':'eos_waiter_phase_proof',**value},semantic=True);value['actual_event']=event
   temporary=waiter_ack.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii');temporary.replace(waiter_ack);previous=packet
 def waiter_guarded():
  try:waiter_worker()
  except BaseException as error:waiter_errors.append(type(error).__name__+': '+str(error))
 waiter_thread=threading.Thread(target=waiter_guarded,name='owned-eos-waiter-proof',daemon=True);waiter_thread.start()
 generate_original=server.StrataEngine.generate
 def generate(self,ids,max_new,sampling,cancel,embeddings=None):
  with lock:counter[0]+=1;call=counter[0];active_calls.add(call)
  with lock:
   if engine_holder[0] is not None and engine_holder[0] is not self:raise ValueError('Different engine cannot borrow fullcache actor controls')
   engine_holder[0]=self
  local.call=call;sent=list(ids);prompt=getattr(local,'prompt',None);generated=[];closed=False;error=None;born=self.gen;pid=self.proc.pid
  emit({'kind':'engine_begin','call':call,'engine_pid':pid,'engine_generation':born,'submitted_ids':sent,'submitted_ids_sha256':token_sha(sent),'rendered_prompt':prompt,'rendered_matches_submitted':bool(prompt and prompt['ids']==sent),'max_new':max_new,'sampling':sampling,'embeddings':embeddings})
  iterator=generate_original(self,ids,max_new,sampling,cancel,embeddings)
  try:
   for token in iterator:
    if type(token) is int:generated.append(token)
    yield token
  except GeneratorExit:
   closed=True;iterator.close();raise
  except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
   last=dict(self.last or {});cancelled=cancel.is_set();rid=last.get('request_id');normal_eos=bool(generated) and generated[-1] in eos and last.get('finish')=='stop' and not cancelled
   if closed and not cancelled and not normal_eos and error is None:error={'type':'GeneratorExit','message':'Noncancelled consumer close without pinned EOS native stop'}
   if self.gen!=born and error is None:error={'type':'EngineGenerationChanged','message':'Native incarnation changed during API request'}
   emit({'kind':'engine_end','call':call,'engine_pid':pid,'engine_generation':born,'rid':rid,'generated_ids':generated,'generated_ids_sha256':token_sha(generated),'engine_last':last,'consumer_closed':closed,'cancelled':cancelled,'terminal_pinned_eos':normal_eos,'error':error,'pinned_eos_ids':eos,'scope':'API yielded IDs/segments only; actual consumed native history separately reconstructed'})
   with lock:active_calls.discard(call)
  local.call=None
 server.Service.encode_prompt=encode;server.popen=popen;server.StrataEngine.generate=generate
 # Observe original close without changing its QUIT/drain/state semantics.
 close_original=server.StrataEngine.close
 def close(self):
  proc=self.proc;error=None
  try:return close_original(self)
  except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
  finally:emit({'kind':'engine_close','engine_pid':getattr(proc,'pid',None),'exit_code':proc.poll() if proc else None,'error':error})
 server.StrataEngine.close=close
 result=None;primary=None;shutdown_error=None
 try:result=server.main()
 except BaseException as exc:primary=exc;raise
 finally:
  phase_stop.set();waiter_stop.set();waiter_thread.join(5);phase_thread.join(timeout=5);control_thread.join(timeout=5);stop_arm.set();arm_thread.join(timeout=2)
  try:
   final_waiter_proof=waiter_observer.proof();Path('/results/waiter-final-proof.json').write_text(json.dumps(final_waiter_proof,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
   if not final_waiter_proof['all_waiter_threads_retired'] or final_waiter_proof['errors']:waiter_errors.append('Final actual waiter proof failed')
   sink.close()
  except BaseException as exc:shutdown_error=exc
  status=sink.status();status['ARM_thread_retired']=not arm_thread.is_alive();status['observer_source_generation']=3;status['fullcache_phase_thread_retired']=not phase_thread.is_alive();status['fullcache_phase_generation']=1;status['fullcache_phase_errors']=list(phase_errors);status['fullcache_control_thread_retired']=not control_thread.is_alive();status['fullcache_control_errors']=list(control_errors);status['waiter_proof_thread_retired']=not waiter_thread.is_alive();status['waiter_proof_errors']=list(waiter_errors)
  if waiter_errors or waiter_thread.is_alive():status.update(passed=False,error=status.get('error') or 'WaiterProofPublisherFailure')
  if control_errors or control_thread.is_alive():status.update(passed=False,error=status.get('error') or 'FullcacheStaleControlFailure')
  if phase_errors:status.update(passed=False,error=status.get('error') or 'FullcachePhaseFailure: '+phase_errors[0])
  if phase_thread.is_alive():status.update(passed=False,error=status.get('error') or 'FullcachePhaseThreadDeadline')
  if arm_thread.is_alive():status.update(passed=False,error=status.get('error') or 'ARMThreadDeadline')
  Path(str(trace)+'.buffered-status.json').write_text(json.dumps(status,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
  if primary is None and shutdown_error is not None:raise shutdown_error
  if primary is None and not status['passed']:raise RuntimeError('Owned buffered trace shutdown failed')
 return result
if __name__=='__main__':raise SystemExit(main())
