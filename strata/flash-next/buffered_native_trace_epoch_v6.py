"""Candidate: binary owned buffers/counters; byte-identical V5 JSON semantics.
Not integrated with any live producer. No event or combined-line filtering.
"""
import hashlib,json,os,threading,time
from pathlib import Path
from buffered_api_trace_sink_v3 import BufferedTrace as Original
from full_cache_trace_process_identity_v1 import pid_identity


def serialize_record(actual,start):
 """Encode every field once; solve only the decimal top-level end-offset width."""
 options={'ensure_ascii':True,'sort_keys':True,'allow_nan':False}
 left={key:value for key,value in actual.items() if key<'trace_end_offset'}
 right={key:value for key,value in actual.items() if key>'trace_end_offset'}
 prefix=((json.dumps(left,**options)[:-1]+', ') if left else '{')+'"trace_end_offset": '
 suffix=(', '+json.dumps(right,**options)[1:] if right else '}')+'\n'
 prefix=prefix.encode('ascii');suffix=suffix.encode('ascii');end=start
 for _ in range(8):
  wanted=start+len(prefix)+len(str(end))+len(suffix)
  if wanted==end:break
  end=wanted
 else:raise ValueError('Exact trace byte-end fixed point failed')
 actual['trace_end_offset']=end
 return prefix+str(end).encode('ascii')+suffix

class BufferedTrace(Original):
 def __init__(self,trace,combined,interval=.1,max_total_bytes=32<<30):
  if Path(trace).exists()or Path(combined).exists():raise ValueError('Fresh exact trace files required')
  if type(max_total_bytes)is not int or not 16<<20<=max_total_bytes<=64<<30:raise ValueError('Declared bounded total text budget required')
  if type(interval)not in(int,float)or not 0<interval<=1:raise ValueError('Bounded trace flush interval0..1second required')
  self.trace=Path(trace);self.combined=Path(combined);self.lock=threading.RLock();self.interval=interval;self.max_total_bytes=max_total_bytes;self.sequence=0;self.record_count=0;self.line_count=0;self.flush_count=0;self.failure=None;self.closed=False;self.stop=threading.Event();self.handles=[]
  try:
   self.trace_handle=self.trace.open('ab');self.handles.append(self.trace_handle);self.combined_handle=self.combined.open('ab');self.handles.append(self.combined_handle)
  except BaseException:
   for handle in self.handles:handle.close()
   raise
  self.trace_offset=0;self.combined_offset=0;self.prefix=hashlib.sha256();self.producer_identity=pid_identity(os.getpid());st=os.fstat(self.trace_handle.fileno());self.trace_file={'device':st.st_dev,'inode':st.st_ino};self.last_marker=None
  self.worker=threading.Thread(target=self._periodic,name='owned-api-trace-flush',daemon=True);self.worker.start()
 def overflow(self,raw,combined_raw):
  if len(raw)>4<<20 or len(combined_raw)>4<<20:raise ValueError('Oversize original trace record; original process must stop')
  target=Path(str(self.trace)+'.overflow.raw');blob=len(raw).to_bytes(8,'little')+len(combined_raw).to_bytes(8,'little')+raw+combined_raw
  if target.exists():raise ValueError('Original text overflow evidence already exists')
  target.write_bytes(blob);self.failure='Owned trace total text budget exhausted; exact first rejected bytes preserved';raise ValueError(self.failure)
 def _emit(self,row,combined_line=None,semantic=False):
  with self.lock:
   self.check();self.sequence+=1;start=self.trace_offset
   actual=dict(row,sequence=self.sequence,epoch=time.time(),producer_identity=self.producer_identity,trace_file=self.trace_file,trace_prefix_sha256_before=self.prefix.hexdigest(),trace_start_offset=start,trace_end_offset=start)
   combined_raw=combined_line.encode('ascii','backslashreplace')+b'\n' if combined_line is not None else b''
   if combined_line is not None:actual.update(combined_start_offset=self.combined_offset,combined_end_offset=self.combined_offset+len(combined_raw),combined_line_index=self.line_count+1)
   if row['kind']=='fullcache_phase_begin':actual['combined_marker']=dict(self.last_marker)
   raw=serialize_record(actual,start)
   if len(raw)>4<<20 or len(combined_raw)>4<<20:raise ValueError('Bounded original trace record required')
   if self.trace_offset+self.combined_offset+len(raw)+len(combined_raw)>self.max_total_bytes-(8<<20)-16:self.overflow(raw,combined_raw)
   if self.trace_handle.write(raw)!=len(raw):raise IOError('Short owned trace write')
   self.trace_offset+=len(raw);self.prefix.update(raw);self.record_count+=1
   if combined_line is not None:
    if self.combined_handle.write(combined_raw)!=len(combined_raw):raise IOError('Short owned combined write')
    self.combined_offset+=len(combined_raw);self.line_count+=1
   if semantic:self._flush()
  return actual
 def marker(self,line):
  try:return self._marker(line)
  except BaseException as error:self.failure=type(error).__name__+': '+str(error);raise
 def _marker(self,line):
  with self.lock:
   self.check();raw=line.encode('ascii')+b'\n'
   if self.trace_offset+self.combined_offset+len(raw)>self.max_total_bytes-(8<<20)-16:self.overflow(b'',raw)
   start=self.combined_offset
   if self.combined_handle.write(raw)!=len(raw):raise IOError('Short owned combined marker write')
   self.combined_offset+=len(raw);self.line_count+=1;self._flush();st=os.fstat(self.combined_handle.fileno());self.last_marker={'start_offset':start,'end_offset':start+len(raw),'line_index':self.line_count,'marker_hex':raw.hex(),'marker_sha256':hashlib.sha256(raw).hexdigest(),'file':{'device':st.st_dev,'inode':st.st_ino}}
 def emit(self,row,combined_line=None,semantic=False):
  try:return self._emit(row,combined_line,semantic)
  except BaseException as error:self.failure=type(error).__name__+': '+str(error);raise
 def anchor(self,row):
  with self.lock:
   self.check();self._flush();start=row['trace_start_offset'];end=row['trace_end_offset']
   if not 0<=start<end or end-start>4<<20:raise ValueError('Bounded original emitted marker required')
   with self.trace.open('rb')as current:current.seek(start);marker=current.read(end-start)
   if len(marker)!=end-start or not marker.endswith(b'\n'):raise ValueError('Original emitted marker absent/truncated')
   return {'end_offset':end,'sequence':row['sequence'],'file':self.trace_file,'producer':self.producer_identity,'marker_hex':marker.hex(),'marker_sha256':hashlib.sha256(marker).hexdigest()}
