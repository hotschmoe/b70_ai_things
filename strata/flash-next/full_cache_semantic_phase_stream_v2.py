"""Live phase events and combined text ranges, keeping complete raw artifacts."""
import copy,hashlib,json,os,stat,threading,time
from pathlib import Path
from incremental_native_semantic_cursor_v1 import Cursor,require,MAX_READ,strict
from full_cache_shared_health_handoff_v6 import process
MAX_PHASE_SEMANTIC=200000
PROTOCOL=('T ','BT ','DONE ','BADM ','BDONE ','PP ','READY','INFO ','ERR','FATAL','YIELDED ','SERR ','SESSION ','SWAIT ','SAVED ','RESTORED ','SBF ','SFD ','LP ','RESUME ','REUSED ','strata batch:','PCL ','FC38 ','FC39 ','SLOT_USM ','MIRROR_USM ')
def semantic(row):return row['kind']!='native_receive' or row['line'].startswith(PROTOCOL)
class FileLines:
 def __init__(self,path,start=0,end=None):
  self.path=Path(path);require(not self.path.is_symlink(),'Original regular combined log required');self.file=self.path.stat();require(stat.S_ISREG(self.file.st_mode),'Actual regular combined log required');self.start=start;self.end=self.file.st_size if end is None else end;require(type(start)is int and type(self.end)is int and 0<=start<=self.end<=self.file.st_size,'Bounded original combined range required')
 def splitlines(self):
  require(not self.path.is_symlink()and(self.path.stat().st_dev,self.path.stat().st_ino)==(self.file.st_dev,self.file.st_ino)and self.path.stat().st_size>=self.end,'Original combined source rotated/truncated')
  with self.path.open('rb')as source:
   source.seek(self.start);remaining=self.end-self.start
   while remaining:
    raw=source.readline(min(4<<20,remaining));require(raw and raw.endswith(b'\n'),'Complete bounded combined line required');remaining-=len(raw);yield raw[:-1].decode('ascii')
class PhaseStream:
 def _initialize(self,out,ack,owned_inspection):
  self.out=Path(out);anchor=ack['trace_anchor'];self.cursor=Cursor(self.out/'api-native-trace.jsonl',anchor);mark=self.cursor.marker;require(mark['index']==ack['index']and mark['name']==ack['name']and mark['engine_pid']==ack['engine_pid']and mark['sequence']==ack['sequence'],'Exact original phase ACK/source marker required')
  self.host_pid=owned_inspection['State']['Pid'];self.host_identity=process(self.host_pid);status=Path('/proc',str(self.host_pid),'status').read_bytes();nspid=next(line for line in status.splitlines()if line.startswith(b'NSpid:'));self.namespace_pids=[int(value)for value in nspid.split()[1:]];require(self.namespace_pids[0]==self.host_pid and self.namespace_pids[-1]==anchor['producer']['pid'],'Actual owned container namespace PID mapping required');self.namespace_status_sha256=hashlib.sha256(status).hexdigest();self.namespace_status_hex=status.hex();require(type(self.host_pid)is int and self.host_pid>0 and self.host_identity['start_ticks']==anchor['producer']['start_ticks'],'Actual owned frontend PID/start namespace association required');self.rows=[mark];self.latest_combined_end=mark['combined_marker']['end_offset'];self.latest_combined_line=mark['combined_marker']['line_index'];self.bytes_scanned=0;self.combined=mark['combined_marker'];self.closed=False;self.lock=threading.RLock()
 def __init__(self,out,ack,owned_inspection):
  self.cursor=None
  try:self._initialize(out,ack,owned_inspection)
  except BaseException:
   if self.cursor is not None:self.cursor.close_failure()
   raise
 def _poll(self):
  current=process(self.host_pid);require(current==self.host_identity,'Current owned producer incarnation changed');rows=self.cursor.poll()
  for row in rows:
   if row['kind']=='fullcache_phase_begin':raise ValueError('Next phase marker before exact prior phase retirement')
   if 'combined_end_offset'in row:self.latest_combined_end=row['combined_end_offset'];self.latest_combined_line=row['combined_line_index']
   if semantic(row):
    require(len(self.rows)<MAX_PHASE_SEMANTIC,'Explicit complete phase semantic bound exceeded');self.rows.append(row)
  self.bytes_scanned=self.cursor.bytes_read;return self.rows
 def poll(self):
  with self.lock:return copy.deepcopy(self._poll())
 def drain_poll(self):
  with self.lock:self._poll()
 def progress(self):
  with self.lock:return {'sequence':self.cursor.sequence,'bytes':self.cursor.offset}
 def action_boundary(self):
  with self.lock:return self.cursor.verify_prefix()
 def combined_range(self,poll=True):
  with self.lock:return self._combined_range(poll)
 def _combined_range(self,poll=True):
  if poll:self.poll()
  end=self.latest_combined_end;return FileLines(self.out/'engine.combined.log',self.combined['end_offset'],end),{'original_first_line':self.combined['line_index']+1,'original_last_line':self.latest_combined_line,'marker':bytes.fromhex(self.combined['marker_hex']).decode('ascii').rstrip('\n')}
 def snapshot(self):
  with self.lock:return copy.deepcopy(self._snapshot())
 def _snapshot(self):
  return {'snapshot_epoch':time.time(),'schema':1,'actual_ACK':self.cursor.anchor,'actual_owned_host_producer':self.host_identity,'actual_NSpid':self.namespace_pids,'actual_proc_status_sha256':self.namespace_status_sha256,'actual_proc_status_hex':self.namespace_status_hex,'last_sequence':self.cursor.sequence,'complete_raw_trace_bytes_consumed':self.cursor.offset,'new_bytes_read':self.cursor.bytes_read,'all_raw_lines_preserved_on_disk':True,'retained_semantic_rows':len(self.rows),'parsed_new_records':self.cursor.parsed_records,'combined_end_offset':self.latest_combined_end,'combined_last_line':self.latest_combined_line,'startup_JSON_materialized':False,'full_prefix_current_action_proof':self.action_boundary(),'between_boundaries_mutation_unobserved':True}
 def freeze(self):
  with self.lock:
   self._poll();rows=copy.deepcopy(self.rows);last=self.cursor.sequence;trace,bounds=self.combined_range(False);snapshot=self.snapshot();require(snapshot['last_sequence']==last,'Semantic tail changed during actual phase freeze')
   return rows,last,trace,bounds,snapshot
 def close(self):
  with self.lock:self.cursor.close_failure();self.closed=True

def ack_binding(ack,marker):
 require(type(ack)is dict and set(ack)=={'index','name','sequence','engine_pid','render_observation','trace_anchor'},'Exact actual phase ACK fields required')
 for key in ('index','name','sequence','engine_pid','render_observation'):require(ack[key]==marker[key]and type(ack[key])is type(marker[key]),'Typed actual marker/ACK changed '+key)
 anchor=ack['trace_anchor'];raw=bytes.fromhex(anchor['marker_hex']);require(hashlib.sha256(raw).hexdigest()==anchor['marker_sha256']and strict(raw)==marker and raw.endswith(b'\n'),'Actual producer marker/source bytes differ');return True

def recollect_rows(path,begin,end):
 """One full raw streaming pass retains only phase semantic records."""
 rows=[]
 with Path(path).open('rb')as source:
  for raw in source:
   require(raw.endswith(b'\n')and len(raw)<=4<<20,'Complete bounded original raw JSON record required');row=strict(raw)
   if begin<row['sequence']<=end and semantic(row):rows.append(row)
 return rows

def phase_range_from_snapshot(out,snapshot):
 out=Path(out);anchor=snapshot['actual_ACK'];mark=strict(bytes.fromhex(anchor['marker_hex']));combined=mark['combined_marker'];end=snapshot['combined_end_offset'];last=snapshot['combined_last_line']
 return FileLines(out/'engine.combined.log',combined['end_offset'],end),{'original_first_line':combined['line_index']+1,'original_last_line':last,'marker':bytes.fromhex(combined['marker_hex']).decode('ascii').rstrip('\n')}

def all_semantic_rows(path):
 rows=[];previous=0
 with Path(path).open('rb')as source:
  for raw in source:
   require(raw.endswith(b'\n')and len(raw)<=4<<20,'Complete bounded original raw trace record required');row=strict(raw);require(type(row['sequence'])is int and row['sequence']==previous+1,'Complete original producer raw sequence required');previous=row['sequence']
   if semantic(row):rows.append(row)
   require(len(rows)<=200000,'Explicit complete semantic record memory bound exceeded')
 return rows

def all_sent_BSTOPs_observed(events):
 from full_cache_shared_terminal_v2 import fields
 for send in events:
  if send['kind']!='native_send' or not send['line'].startswith('BSTOP '):continue
  parts=send['line'].split();tags=fields(send['line']);slot=int(parts[1]);rid=int(tags['rid']);generation=int(tags['slotgen']);matches=[]
  for event in events:
   if event['kind']!='native_receive' or event['engine_pid']!=send['engine_pid'] or event['sequence']<=send['sequence'] or not event['line'].startswith('FC39 '):continue
   row=strict(event['line'][5:])
   if row.get('kind')=='native_lifetime'and row['phase']=='BSTOP_applied'and(row['rid'],row['slotgen'],row['slot'])==(rid,generation,slot):matches.append(event)
  require(len(matches)==1,'Actual sent current-owner BSTOP metadata apply pending/ambiguous')
 return True

def snapshot_binding(out,snapshot,ack,owned_sample=None):
 out=Path(out);require(snapshot['schema']==1 and type(snapshot['schema'])is int and snapshot['actual_ACK']==ack['trace_anchor']and snapshot['all_raw_lines_preserved_on_disk']is True and snapshot['startup_JSON_materialized']is False,'Actual original phase source cursor scope differs')
 raw=bytes.fromhex(snapshot['actual_proc_status_hex']);require(hashlib.sha256(raw).hexdigest()==snapshot['actual_proc_status_sha256'],'Original owned PID namespace status bytes differ');pids=[int(v)for v in next(line for line in raw.splitlines()if line.startswith(b'NSpid:')).split()[1:]];require(pids==snapshot['actual_NSpid']and pids[0]==snapshot['actual_owned_host_producer']['pid']and pids[-1]==ack['trace_anchor']['producer']['pid']and snapshot['actual_owned_host_producer']['start_ticks']==ack['trace_anchor']['producer']['start_ticks'],'Exact original host/container producer namespace differs')
 if owned_sample is not None:
  inspection=owned_sample['ownership_receipt']['inspection'];require(inspection['State']['Running']is True and inspection['State']['Pid']==pids[0]==owned_sample['container_root_host_pid'],'Original independently sampled actual container PID differs');identities=[r for r in owned_sample['owned_process_identities']if r['pid']==pids[0]];require(len(identities)==1 and identities[0]['start_ticks']==snapshot['actual_owned_host_producer']['start_ticks'],'Actual original owned sampled PID/start differs')
  statuses=[r for r in owned_sample['raw_receipts']if Path(r['path']).name=='status-'+str(pids[0])+'.raw'];require(len(statuses)==1,'Actual independently captured owner namespace status absent');original=Path(statuses[0]['path']);require(not original.is_symlink(),'Original sample status alias refused');sample_raw=original.read_bytes();require(len(sample_raw)==statuses[0]['bytes']and hashlib.sha256(sample_raw).hexdigest()==statuses[0]['sha256'],'Original sampled namespace bytes differ');sample_pids=[int(v)for v in next(line for line in sample_raw.splitlines()if line.startswith(b'NSpid:')).split()[1:]];require(sample_pids==pids,'Original sampled actor/trace namespace mapping differs')
 proof=snapshot['full_prefix_current_action_proof'];require(proof['fresh_full_prefix_read']is True and proof['file']==ack['trace_anchor']['file']and proof['producer']==ack['trace_anchor']['producer']and type(proof['bytes'])is int and proof['bytes']==snapshot['complete_raw_trace_bytes_consumed'],'Actual phase original complete byte prefix differs');digest=hashlib.sha256();remaining=proof['bytes'];path=out/'api-native-trace.jsonl';require(not path.is_symlink(),'Original raw trace alias refused')
 with path.open('rb')as source:
  while remaining:
   chunk=source.read(min(MAX_READ,remaining));require(chunk,'Original raw prefix EOF');digest.update(chunk);remaining-=len(chunk)
 require(digest.hexdigest()==proof['sha256'],'Original consumed phase raw byte prefix changed');return True
