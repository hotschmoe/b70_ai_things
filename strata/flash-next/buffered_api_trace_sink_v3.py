"""Fixed owned trace handles; exact records, bounded visibility and fail-closed drain."""
import json,threading,time
from pathlib import Path

class BufferedTrace:
 def __init__(self,trace,combined,interval=0.1):
  if type(interval) not in (int,float) or not 0<interval<=1:raise ValueError('Bounded trace flush interval0..1second required')
  self.trace=Path(trace);self.combined=Path(combined);self.lock=threading.RLock();self.interval=interval;self.sequence=0;self.record_count=0;self.line_count=0;self.flush_count=0;self.failure=None;self.closed=False;self.stop=threading.Event();self.handles=[]
  try:
   self.trace_handle=self.trace.open('a',encoding='ascii');self.handles.append(self.trace_handle);self.combined_handle=self.combined.open('a',encoding='ascii');self.handles.append(self.combined_handle)
  except BaseException:
   for handle in self.handles:handle.close()
   raise
  self.worker=threading.Thread(target=self._periodic,name='owned-api-trace-flush',daemon=True);self.worker.start()
 def check(self):
  if self.failure is not None:raise RuntimeError('Owned buffered trace failed: '+self.failure)
  if self.closed:raise RuntimeError('Owned buffered trace already closed')
 def _flush(self):
  try:
   self.trace_handle.flush();self.combined_handle.flush();self.flush_count+=1
  except BaseException as exc:self.failure=type(exc).__name__+': '+str(exc);raise
 def flush(self):
  with self.lock:self.check();self._flush()
 def emit(self,row,combined_line=None,semantic=False):
  with self.lock:
   self.check();self.sequence+=1;actual=dict(row,sequence=self.sequence,epoch=time.time())
   try:
    self.trace_handle.write(json.dumps(actual,ensure_ascii=True,sort_keys=True)+'\n');self.record_count+=1
    if combined_line is not None:self.combined_handle.write(combined_line.encode('ascii','backslashreplace').decode('ascii')+'\n');self.line_count+=1
    if semantic:self._flush()
   except BaseException as exc:self.failure=type(exc).__name__+': '+str(exc);raise
  return actual
 def marker(self,line):
  with self.lock:
   self.check()
   try:self.combined_handle.write(line+'\n');self.line_count+=1;self._flush()
   except BaseException as exc:self.failure=type(exc).__name__+': '+str(exc);raise
 def _periodic(self):
  while not self.stop.wait(self.interval):
   try:self.flush()
   except BaseException:return
 def close(self):
  self.stop.set();self.worker.join(timeout=max(2,self.interval*4))
  if self.worker.is_alive():
   self.failure='FlushThreadDeadline: owned periodic writer did not retire'
   raise RuntimeError(self.failure)
  with self.lock:
   if self.closed:
    if self.failure is not None:raise RuntimeError('Owned buffered trace failed: '+self.failure)
    return self.status()
   if self.failure is None:
    try:self._flush()
    except BaseException:pass
   for handle in self.handles:
    try:handle.close()
    except BaseException as exc:self.failure=self.failure or type(exc).__name__+': '+str(exc)
   self.closed=True;status=self.status()
  if self.failure is not None:raise RuntimeError('Owned buffered trace failed: '+self.failure)
  return status
 def status(self):return {'schema':3,'passed':self.closed and self.failure is None,'records':self.record_count,'combined_lines':self.line_count,'flushes':self.flush_count,'flush_interval_seconds':self.interval,'owned_handles':2,'closed':self.closed,'periodic_thread_retired':not self.worker.is_alive(),'error':self.failure,'physical_fsync_durability_qualified':False}
