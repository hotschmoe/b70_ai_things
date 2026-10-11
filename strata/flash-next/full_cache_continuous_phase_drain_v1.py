"""Owned bounded incremental reader during HTTP, preserving raw/source sequences."""
import hashlib,json,math,os,threading,time
from pathlib import Path
from full_cache_trace_process_identity_v1 import pid_identity

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class ContinuousDrain:
 def __init__(self,stream,output,write):
  self.stream=stream;self.output=Path(output);self.write=write;self.stop=threading.Event();self.lock=threading.Lock();self.failure=None;self.polls=0;self.entered=None;self.exited=None;self.started=time.time();self.owner=pid_identity(os.getpid());self.thread=threading.Thread(target=self._run,name='owned-fullcache-phase-raw-drain',daemon=False);self.finished=None;self.thread.start()
 def _run(self):
  self.entered=time.time()
  try:
   while not self.stop.is_set():
    before=self.stream.progress();self.stream.drain_poll();self.polls+=1;after=self.stream.progress()
    if after==before:self.stop.wait(.01)
  except BaseException as error:
   with self.lock:self.failure=type(error).__name__+': '+str(error)
  finally:self.exited=time.time()
 def check(self):
  with self.lock:require(self.failure is None,'Actual continuous raw phase drain failed: '+str(self.failure))
 def require_sequence(self,sequence):
  self.check();require(type(sequence)is int and sequence>0,'Exact actual producer proof sequence required');require(self.stream.progress()['sequence']>=sequence,'Actual continuous cursor has not reached original waiter proof ACK')
 def finish(self):
  self.stop.set();self.thread.join();self.finished=time.time();row={'schema':1,'owner':self.owner,'name':self.thread.name,'thread_ident':self.thread.ident,'thread_native_id':self.thread.native_id,'started_epoch':self.started,'entered_epoch':self.entered,'exited_epoch':self.exited,'finished_epoch':self.finished,'actual_thread_started':self.thread.ident is not None,'actual_thread_retired':not self.thread.is_alive(),'error':self.failure,'poll_count':self.polls,'last_progress':self.stream.progress(),'source_sha256':sha(__file__),'raw_lines_skipped':False,'terminal_deadline_extended':False};self.write(self.output/'continuous-drain-receipt.json',row);return row
 def __enter__(self):return self
 def __exit__(self,kind,value,tb):
  try:
   if self.finished is None:self.finish()
   if kind is None:self.check()
  finally:self.stream.close()
  return False

def binding(row,ack,snapshot,actor_pid):
 require(type(row)is dict and row['schema']==1 and type(row['schema'])is int and row['source_sha256']==sha(__file__),'Exact continuous drain source/receipt required');require(type(actor_pid)is int and set(row['owner'])=={'pid','start_ticks'}and row['owner']['pid']==actor_pid and type(row['owner']['pid'])is int and type(row['owner']['start_ticks'])is int,'Original actor owns actual reader thread');require(row['actual_thread_started']is True and row['actual_thread_retired']is True and row['error']is None and type(row['thread_ident'])is int and type(row['thread_native_id'])is int and row['name']=='owned-fullcache-phase-raw-drain','Original successful thread lifetime required');require(type(row['poll_count'])is int and row['poll_count']>0 and row['raw_lines_skipped']is False and row['terminal_deadline_extended']is False,'Original bounded reader semantics changed')
 epochs=[row[k]for k in('started_epoch','entered_epoch','exited_epoch','finished_epoch')];require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in epochs)and epochs==sorted(epochs),'Original actual drain thread chronology differs');require(type(row['last_progress']['sequence'])is int and ack['sequence']<=row['last_progress']['sequence']<=snapshot['last_sequence']and type(row['last_progress']['bytes'])is int and row['last_progress']['bytes']<=snapshot['complete_raw_trace_bytes_consumed'],'Original reader/final atomic snapshot sequence differs');return {'actual_continuous_reader_retired_before_atomic_phase_freeze':True,'raw_evidence_or_deadline_waived':False}
