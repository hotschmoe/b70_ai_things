"""NEW trace metadata; exact old event fields and all combined lines retained."""
import hashlib,json,os,time
from pathlib import Path
from buffered_api_trace_sink_v3 import BufferedTrace as Original
from full_cache_trace_process_identity_v1 import pid_identity
class BufferedTrace(Original):
 def __init__(self,trace,combined,interval=.1,max_total_bytes=32<<30):
  if Path(trace).exists()or Path(combined).exists():raise ValueError('Fresh exact trace files required')
  if type(max_total_bytes)is not int or not 16<<20<=max_total_bytes<=64<<30:raise ValueError('Declared bounded total text budget required')
  self.max_total_bytes=max_total_bytes
  super().__init__(trace,combined,interval);self.prefix=hashlib.sha256();self.producer_identity=pid_identity(os.getpid());st=os.fstat(self.trace_handle.fileno());self.trace_file={'device':st.st_dev,'inode':st.st_ino};self.last_marker=None
 def overflow(self,raw,combined_raw):
  # Preserve the exact bounded first rejected record and source line. The
  # reserved failure record is included in the declared total budget.
  if len(raw)>4<<20 or len(combined_raw)>4<<20:raise ValueError('Oversize original trace record; original process must stop')
  target=Path(str(self.trace)+'.overflow.raw');blob=len(raw).to_bytes(8,'little')+len(combined_raw).to_bytes(8,'little')+raw+combined_raw
  if target.exists():raise ValueError('Original text overflow evidence already exists')
  target.write_bytes(blob);self.failure='Owned trace total text budget exhausted; exact first rejected bytes preserved';raise ValueError(self.failure)
 def _emit(self,row,combined_line=None,semantic=False):
  with self.lock:
   self.check();self.sequence+=1;start=self.trace_handle.tell();actual=dict(row,sequence=self.sequence,epoch=time.time(),producer_identity=self.producer_identity,trace_file=self.trace_file,trace_prefix_sha256_before=self.prefix.hexdigest(),trace_start_offset=start,trace_end_offset=start)
   if combined_line is not None:
    actual['combined_start_offset']=self.combined_handle.tell();actual['combined_end_offset']=self.combined_handle.tell()+len(combined_line.encode('ascii','backslashreplace'))+1;actual['combined_line_index']=self.line_count+1
   if row['kind']=='fullcache_phase_begin':actual['combined_marker']=dict(self.last_marker)
   for _ in range(8):
    raw=(json.dumps(actual,ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n').encode('ascii');end=start+len(raw)
    if actual['trace_end_offset']==end:break
    actual['trace_end_offset']=end
   else:raise ValueError('Exact trace byte-end fixed point failed')
   combined_raw=(combined_line.encode('ascii','backslashreplace')+b'\n') if combined_line is not None else b''
   if len(raw)>4<<20 or len(combined_raw)>4<<20:raise ValueError('Bounded original trace record required')
   if self.trace_handle.tell()+self.combined_handle.tell()+len(raw)+len(combined_raw)>self.max_total_bytes-(8<<20)-16:self.overflow(raw,combined_raw)
   self.trace_handle.write(raw.decode('ascii'));self.prefix.update(raw);self.record_count+=1
   if combined_line is not None:self.combined_handle.write(combined_line.encode('ascii','backslashreplace').decode('ascii')+'\n');self.line_count+=1
   if semantic:self._flush()
  return actual
 def marker(self,line):
  with self.lock:
   self.check();raw=line.encode('ascii')+b'\n'
   if self.trace_handle.tell()+self.combined_handle.tell()+len(raw)>self.max_total_bytes-(8<<20)-16:self.overflow(b'',raw)
   start=self.combined_handle.tell();self.combined_handle.write(raw.decode('ascii'));self.line_count+=1;self._flush();self.last_marker={'start_offset':start,'end_offset':start+len(raw),'line_index':self.line_count,'marker_hex':raw.hex(),'marker_sha256':hashlib.sha256(raw).hexdigest(),'file':{'device':os.fstat(self.combined_handle.fileno()).st_dev,'inode':os.fstat(self.combined_handle.fileno()).st_ino}}
 def emit(self,row,combined_line=None,semantic=False):
  try:return self._emit(row,combined_line,semantic)
  except BaseException as error:self.failure=type(error).__name__+': '+str(error);raise
 def anchor(self,row):
  with self.lock:
   self.check();self._flush();start=row['trace_start_offset'];end=row['trace_end_offset']
   if not 0<=start<end or end-start>4<<20:raise ValueError('Bounded original emitted marker required')
   with Path(self.trace).open('rb')as current:current.seek(start);marker=current.read(end-start)
   if len(marker)!=end-start or not marker.endswith(b'\n'):raise ValueError('Original emitted marker absent/truncated')
   return {'end_offset':end,'sequence':row['sequence'],'file':self.trace_file,'producer':self.producer_identity,'marker_hex':marker.hex(),'marker_sha256':hashlib.sha256(marker).hexdigest()}
