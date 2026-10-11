"""Append cursor successor: metadata suspicion requires current byte proof."""
import hashlib,os,time
from incremental_native_semantic_cursor_v1 import Cursor as Original,require,identity,MAX_READ,MAX_RECORD,strict
MAX_FALLBACKS=1024

def stat_row(st):
 return {'device':st.st_dev,'inode':st.st_ino,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns}
class Cursor(Original):
 def _initialize(self,path,anchor):
  super()._initialize(path,anchor);self.metadata_fallbacks=[];self.metadata_fallback_bytes=0;self.metadata_fallback_seconds=0.
 def _metadata_proof(self,before):
  require(len(self.metadata_fallbacks)<MAX_FALLBACKS,'Explicit metadata proof record bound exceeded')
  started=time.time();previous=stat_row(self.last_stat);observed=stat_row(before);proof=self.verify_prefix()
  # Include the consumed incomplete record, which verify_prefix deliberately
  # does not treat as a complete source record/action marker.
  with self.path.open('rb')as current:
   require(identity(os.fstat(current.fileno()))==self.file,'Metadata fallback original source rotated');current.seek(self.offset);pending=current.read(len(self.pending));require(pending==self.pending,'Earlier consumed partial trace bytes mutated')
  after=self.path.stat();fd=os.fstat(self.handle.fileno());require(not self.path.is_symlink()and identity(after)==identity(fd)==self.file and after.st_size>=self.handle.tell(),'Metadata proof source rotated/truncated')
  finished=time.time();row={'label':'same_size_timestamp_suspicion_current_byte_proof','before_stat':observed,'previous_stat':previous,'after_stat':stat_row(after),'started_epoch':started,'finished_epoch':finished,'seconds':finished-started,'complete_consumed_prefix':proof,'pending_bytes':len(pending),'pending_sha256':hashlib.sha256(pending).hexdigest(),'pending_current_bytes_equal':True,'metadata_alone_trusted':False,'actual_inplace_mutation_cause_claimed':False}
  self.metadata_fallbacks.append(row);self.metadata_fallback_bytes+=proof['bytes']+len(pending);self.metadata_fallback_seconds+=row['seconds']
 def poll(self):
  require(not self.closed,'Retired cursor cannot resume');before=self.path.stat();fd=os.fstat(self.handle.fileno());require(not self.path.is_symlink()and identity(before)==identity(fd)==self.file and before.st_size>=self.offset+len(self.pending),'Trace rotation/truncation refused')
  if before.st_size==self.last_stat.st_size and (before.st_mtime_ns,before.st_ctime_ns)!=(self.last_stat.st_mtime_ns,self.last_stat.st_ctime_ns):self._metadata_proof(before)
  available=before.st_size-self.handle.tell();require(available>=0,'Trace shorter than actual cursor');raw=self.handle.read(min(MAX_READ,available));self.bytes_read+=len(raw);buffer=self.pending+raw;parts=buffer.split(b'\n');self.pending=parts.pop();require(len(self.pending)<=MAX_RECORD,'Bounded partial trace record required');rows=[]
  for part in parts:
   line=part+b'\n';require(0<len(part)<=MAX_RECORD,'Bounded nonempty complete trace row required');row=strict(part);require(type(row['sequence'])is int and row['sequence']==self.sequence+1,'Exact contiguous producer sequence required');require(type(row['producer_identity']['pid'])is int and type(row['producer_identity']['start_ticks'])is int and row['producer_identity']==self.owner and all(type(v)is int for v in row['trace_file'].values())and row['trace_file']==self.file and type(row['trace_start_offset'])is int and row['trace_start_offset']==self.offset and type(row['trace_end_offset'])is int and row['trace_end_offset']==self.offset+len(line),'Exact current source/record byte boundary required');self.last_line_start=self.offset;self.last_line=line;self.expected_prefix_before=row['trace_prefix_sha256_before'];self.offset+=len(line);self.sequence=row['sequence'];self.parsed_records+=1;rows.append(row)
  self.last_stat=self.path.stat();require(identity(self.last_stat)==self.file and self.last_stat.st_size>=self.handle.tell(),'Trace changed during append read');return rows
 def metadata_proof_receipt(self):
  return {'schema':1,'fallback_count':len(self.metadata_fallbacks),'fallback_bytes_read':self.metadata_fallback_bytes,'fallback_seconds':self.metadata_fallback_seconds,'fallbacks':list(self.metadata_fallbacks),'stat_only_content_trusted':False,'actual_failure_cause_claimed':False,'full_prefix_action_and_final_checks_unchanged':True}

def metadata_receipt_binding(path,snapshot):
 """Recollect fallback prefix checkpoints in one bounded current-byte pass."""
 import math
 row=snapshot['metadata_timestamp_proof'];fallbacks=row['fallbacks'];require(type(row['schema'])is int and row['schema']==1 and type(row['fallback_count'])is int and row['fallback_count']==len(fallbacks)<=MAX_FALLBACKS and row['stat_only_content_trusted']is False and row['actual_failure_cause_claimed']is False and row['full_prefix_action_and_final_checks_unchanged']is True,'Exact metadata proof receipt/scope required')
 file=snapshot['actual_ACK']['file'];owner=snapshot['actual_ACK']['producer'];expected={};cost=0.;total=0;last=0.
 def epoch(v):require(type(v)in(int,float)and math.isfinite(v)and v>=0,'Finite typed metadata proof epoch/cost required');return v
 for item in fallbacks:
  require(item['label']=='same_size_timestamp_suspicion_current_byte_proof'and item['pending_current_bytes_equal']is True and item['metadata_alone_trusted']is False and item['actual_inplace_mutation_cause_claimed']is False,'Actual metadata fallback scope differs')
  for name in('previous_stat','before_stat','after_stat'):
   st=item[name];require(set(st)=={'device','inode','bytes','mtime_ns','ctime_ns'}and all(type(v)is int and v>=0 for v in st.values())and {'device':st['device'],'inode':st['inode']}==file,'Typed original fallback file identity differs')
  a=item['previous_stat'];b=item['before_stat'];z=item['after_stat'];require(a['bytes']==b['bytes']<=z['bytes']and(a['mtime_ns'],a['ctime_ns'])!=(b['mtime_ns'],b['ctime_ns']),'Actual original metadata suspicion differs')
  begin=epoch(item['started_epoch']);end=epoch(item['finished_epoch']);seconds=epoch(item['seconds']);require(last<=begin<=end and seconds==end-begin,'Actual original fallback chronology/cost differs');last=end;cost+=seconds
  proof=item['complete_consumed_prefix'];n=proof['bytes'];pending=item['pending_bytes'];require(type(n)is int and type(pending)is int and 0<=pending<=MAX_RECORD and 0<n<=n+pending<=b['bytes']and n<=snapshot['complete_raw_trace_bytes_consumed']and proof['file']==file and proof['producer']==owner and proof['fresh_full_prefix_read']is True and proof['between_boundary_mutation_unobserved']is True,'Actual original fallback byte/owner scope differs')
  require(type(proof['producer'])is dict and set(proof['producer'])=={'pid','start_ticks'}and all(type(v)is int for v in proof['producer'].values())and type(proof['file'])is dict and set(proof['file'])=={'device','inode'}and all(type(v)is int for v in proof['file'].values()),'Typed original fallback owner/file required')
  require(type(proof['sequence'])is int and owner['pid']>0 and snapshot['actual_ACK']['sequence']<=proof['sequence']<=snapshot['last_sequence'],'Actual original fallback sequence differs')
  expected.setdefault(n,set()).add(proof['sha256']);total+=n+pending
 require(type(row['fallback_bytes_read'])is int and row['fallback_bytes_read']==total and epoch(row['fallback_seconds'])==cost,'Actual original fallback count/byte/cost aggregate differs')
 path=__import__('pathlib').Path(path);before=path.stat();require(not path.is_symlink()and identity(before)==file and before.st_size>=snapshot['complete_raw_trace_bytes_consumed'],'Current fallback original source rotated/truncated')
 with path.open('rb')as source:
  require(identity(os.fstat(source.fileno()))==file,'Current fallback descriptor differs');digest=hashlib.sha256();offset=0
  for limit in sorted(expected):
   while offset<limit:
    chunk=source.read(min(MAX_READ,limit-offset));require(chunk,'Current fallback prefix EOF');digest.update(chunk);offset+=len(chunk)
   require(expected[limit]=={digest.hexdigest()},'Current original fallback prefix bytes mutated')
  for item in fallbacks:
   source.seek(item['complete_consumed_prefix']['bytes']);pending=source.read(item['pending_bytes']);require(len(pending)==item['pending_bytes']and hashlib.sha256(pending).hexdigest()==item['pending_sha256'],'Current original fallback partial bytes mutated')
 after=path.stat();require(not path.is_symlink()and identity(after)==file and after.st_size>=snapshot['complete_raw_trace_bytes_consumed'],'Current fallback source retired/rotated/truncated')
 return {'actual_metadata_fallbacks_recollected':len(fallbacks),'fallback_bytes_read':total,'original_fallback_seconds':cost,'metadata_only_trust_used':False,'cause_claimed':False}
