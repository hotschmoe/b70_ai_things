"""Bounded live append cursor; raw history remains complete on disk."""
import hashlib,json,os,stat
from pathlib import Path
from serial37_canonical_json_v3 import unique_object,finite_constant
def strict(raw):return json.loads(raw,object_pairs_hook=unique_object,parse_constant=finite_constant)
MAX_RECORD=4<<20
MAX_READ=8<<20

def require(ok,msg):
 if not ok:raise ValueError(msg)
def identity(st):return {'device':st.st_dev,'inode':st.st_ino}
class Cursor:
 def _initialize(self,path,anchor):
  self.path=Path(path);require(type(anchor['file'])is dict and set(anchor['file'])=={'device','inode'}and all(type(v)is int and v>=0 for v in anchor['file'].values()),'Typed exact trace file identity required');require(not self.path.is_symlink()and self.path==self.path.resolve(),'Original regular trace path required');self.handle=self.path.open('rb');st=os.fstat(self.handle.fileno());require(stat.S_ISREG(st.st_mode)and identity(st)==anchor['file'],'Exact current trace inode/device required');self.file=identity(st);self.owner=anchor['producer'];require(type(self.owner['pid'])is int and self.owner['pid']>0 and type(self.owner['start_ticks'])is int and self.owner['start_ticks']>=0,'Typed actual producer incarnation required')
  self.offset=anchor['end_offset'];self.sequence=anchor['sequence'];require(type(self.offset)is int and 0<=self.offset<=st.st_size and type(self.sequence)is int and self.sequence>0,'Typed complete ACK boundary required');require(type(anchor['marker_hex'])is str,'Original marker bytes required');marker=bytes.fromhex(anchor['marker_hex']);require(marker.endswith(b'\n')and len(marker)<=MAX_RECORD,'Complete bounded original marker required');start=self.offset-len(marker);require(start>=0,'Marker begins before trace');self.handle.seek(start);require(self.handle.read(len(marker))==marker and hashlib.sha256(marker).hexdigest()==anchor['marker_sha256'],'Exact original ACK marker bytes changed');parsed=strict(marker);require(type(parsed['trace_start_offset'])is int and parsed['trace_start_offset']==start and type(parsed['sequence'])is int and parsed['sequence']==self.sequence and parsed['producer_identity']==self.owner and parsed['trace_file']==self.file and type(parsed['trace_end_offset'])is int and parsed['trace_end_offset']==self.offset,'Original marker/source offset differs');require(parsed['kind']=='fullcache_phase_begin'and type(parsed['index'])is int and parsed['index']>0 and type(parsed['engine_pid'])is int and parsed['engine_pid']>0,'Exact actual semantic phase marker required');self.marker=parsed;self.handle.seek(self.offset);self.pending=b'';self.records=[];self.last_stat=st;self.closed=False;self.bytes_read=0;self.parsed_records=0;self.anchor=dict(anchor);self.last_line_start=start;self.last_line=marker;self.expected_prefix_before=parsed['trace_prefix_sha256_before']
 def __init__(self,path,anchor):
  self.handle=None
  try:self._initialize(path,anchor)
  except BaseException:
   if self.handle is not None:self.handle.close()
   raise
 def poll(self):
  require(not self.closed,'Retired cursor cannot resume');before=self.path.stat();fd=os.fstat(self.handle.fileno());require(not self.path.is_symlink()and identity(before)==identity(fd)==self.file and before.st_size>=self.offset+len(self.pending),'Trace rotation/truncation refused')
  require(before.st_size!=self.last_stat.st_size or (before.st_mtime_ns,before.st_ctime_ns)==(self.last_stat.st_mtime_ns,self.last_stat.st_ctime_ns),'In-place mutation without append refused')
  available=before.st_size-self.handle.tell();require(available>=0,'Trace shorter than actual cursor');raw=self.handle.read(min(MAX_READ,available));self.bytes_read+=len(raw);buffer=self.pending+raw;parts=buffer.split(b'\n');self.pending=parts.pop();require(len(self.pending)<=MAX_RECORD,'Bounded partial trace record required');rows=[]
  for part in parts:
   line=part+b'\n';require(0<len(part)<=MAX_RECORD,'Bounded nonempty complete trace row required');row=strict(part);require(type(row['sequence'])is int and row['sequence']==self.sequence+1,'Exact contiguous producer sequence required');require(type(row['producer_identity']['pid'])is int and type(row['producer_identity']['start_ticks'])is int and row['producer_identity']==self.owner and all(type(v)is int for v in row['trace_file'].values())and row['trace_file']==self.file and type(row['trace_start_offset'])is int and row['trace_start_offset']==self.offset and type(row['trace_end_offset'])is int and row['trace_end_offset']==self.offset+len(line),'Exact current source/record byte boundary required');self.last_line_start=self.offset;self.last_line=line;self.expected_prefix_before=row['trace_prefix_sha256_before'];self.offset+=len(line);self.sequence=row['sequence'];self.parsed_records+=1;rows.append(row)
  self.last_stat=self.path.stat();require(identity(self.last_stat)==self.file and self.last_stat.st_size>=self.handle.tell(),'Trace changed during append read');return rows
 def verify_prefix(self):
  """Fresh full prefix bytes at an action boundary, never a saved PASS."""
  require(not self.path.is_symlink()and identity(self.path.stat())==self.file,'Original prefix source changed');limit=self.last_line_start;digest=hashlib.sha256();count=0
  with self.path.open('rb')as source:
   while count<limit:
    raw=source.read(min(MAX_READ,limit-count));require(raw,'Unexpected prefix EOF');digest.update(raw);count+=len(raw)
  require(digest.hexdigest()==self.expected_prefix_before,'Earlier original trace prefix mutation refused');digest.update(self.last_line)
  with self.path.open('rb')as current:
   current.seek(self.last_line_start);require(current.read(len(self.last_line))==self.last_line,'Current action marker/tail changed')
  return {'bytes':count+len(self.last_line),'sha256':digest.hexdigest(),'file':self.file,'producer':self.owner,'sequence':self.sequence,'fresh_full_prefix_read':True,'between_boundary_mutation_unobserved':True}
 def iter_retired(self,producer_terminal,original_eof):
  require(producer_terminal is True and original_eof is True,'Actual producer terminal and original EOF required')
  while self.handle.tell()<self.path.stat().st_size:
   for row in self.poll():yield row
  require(self.pending==b''and self.path.stat().st_size==self.offset,'Retired EOF cannot contain partial or unconsumed bytes');proof=self.verify_prefix();self.handle.close();self.closed=True;self.retirement_proof=proof
 def retire(self,producer_terminal,original_eof):
  count=0
  for row in self.iter_retired(producer_terminal,original_eof):count+=1
  return {'drained_records':count,'proof':self.retirement_proof,'unbounded_tail_materialized':False}
 def close_failure(self):
  self.handle.close();self.closed=True
