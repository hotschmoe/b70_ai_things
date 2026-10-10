"""Explicit process-local fresh-byte pack epoch prototype, never persisted trust.
Future consumers must pass this object deliberately; frozen SHA helpers are not
patched. Every semantic consumer still checks its own current expected roster.
"""
import hashlib,os,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical

def require(ok,message):
 if not ok:raise ValueError(message)
def stat5(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def current_row(path):
 path=Path(path).resolve();before=stat5(path);h=hashlib.sha256();count=0
 with path.open('rb') as handle:
  descriptor=os.fstat(handle.fileno());require([descriptor.st_dev,descriptor.st_ino,descriptor.st_size,descriptor.st_mtime_ns,descriptor.st_ctime_ns]==before,'Pack path/descriptor identity changed')
  while True:
   block=handle.read(8<<20)
   if not block:break
   h.update(block);count+=len(block)
  after_fd=os.fstat(handle.fileno())
 after=stat5(path);require(before==after==[after_fd.st_dev,after_fd.st_ino,after_fd.st_size,after_fd.st_mtime_ns,after_fd.st_ctime_ns] and count==before[2],'Pack mutated during complete byte read')
 return {'path':str(path),'sha256':h.hexdigest(),'bytes':count,'stat_before':before,'stat_after':after}
class PackEpoch:
 def __init__(self,roster,max_seconds=900):
  require(type(max_seconds)is int and 1<=max_seconds<=10800,'Bounded operation duration required');self.owner=os.getpid();self.started=time.monotonic();self.max_seconds=max_seconds;self.phase='admission';self.calls=0;self.boundaries=[];self.expected={}
  for path,digest in roster:
   path=str(Path(path).resolve());require(path not in self.expected,'Duplicate/aliased pack path');require(type(digest)is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Exact expected pack SHA required');self.expected[path]=digest
  require(1<=len(self.expected)<=32,'Bounded complete explicit pack roster required');self.initial=self._fresh('admission_start')
 def _owned(self):
  require(os.getpid()==self.owner and time.monotonic()-self.started<=self.max_seconds,'Witness ownership/operation lifetime differs')
 def _fresh(self,label):
  self._owned();start=time.time();rows=[current_row(path) for path in sorted(self.expected)]
  require(all(row['sha256']==self.expected[row['path']] for row in rows),'Current pack bytes differ expected source receipt')
  if hasattr(self,'initial'):require(canonical(rows)==canonical(self.initial),'Pack identity changed across fresh-byte boundaries')
  self.boundaries.append({'label':label,'started_epoch':start,'finished_epoch':time.time(),'rows':rows});return rows
 def require_roster(self,roster):
  self._owned();current={}
  for path,digest in roster:
   path=str(Path(path).resolve());require(path not in current,'Duplicate current pack roster');current[path]=digest
  require(canonical(current)==canonical(self.expected),'Semantic caller current pack roster differs');return True
 def digest_for(self,path,expected):
  self._owned();require(self.phase in ('admission','sealed'),'Semantic digest consumption after closure refused');path=str(Path(path).resolve());require(path in self.expected and expected==self.expected[path],'Unexpected/stale semantic caller pack file');self.calls+=1
  # This is an in-memory result from this operation's actual full byte read.
  # It is not a new current-byte proof. Seal MUST re-read every byte predevice.
  return self.expected[path]
 def seal_predevice(self):
  self._owned();require(self.phase=='admission','Duplicate predevice seal');self._fresh('predevice_complete_byte_recheck');self.phase='sealed';return self.boundaries[-1]
 def finalize(self):
  self._owned();require(self.phase=='sealed','Fresh predevice seal required before final proof');self._fresh('postoperation_complete_byte_recheck');self.phase='closed'
  return {'schema':1,'owner_pid':self.owner,'semantic_digest_calls':self.calls,'unique_pack_files':len(self.expected),'boundaries':self.boundaries,'saved_digest_imported':False,'stat_only_validation':False,'between_boundaries_mutation_unobserved':True,'actual_GPU_execution_proven':False,'full_semantic_admission_proven':False,'optimization_integrated_into_runtime':False}

# Explicit future consumer adapter for validate_prepared's pack loop only.
# All other generation/upload/layout/source/health/identity gates stay required.
def consume_prepared_pack(prepared,epoch):
 from serial37_canonical_json_v3 import read_unique
 require(isinstance(epoch,PackEpoch),'Explicit current-process PackEpoch required; no saved witness import');receipt=Path(prepared['pack_receipt']);before=stat5(receipt);raw=receipt.read_bytes();require(hashlib.sha256(raw).hexdigest()==prepared['pack_receipt_sha256'],'Current pack intake receipt changed');value=read_unique(receipt);require(stat5(receipt)==before and receipt.read_bytes()==raw,'Pack intake receipt changed while read')
 roster=[]
 for name,identity in value['RESULT']['files'].items():
  relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Invalid relative pack member');roster.append((Path(prepared['pack'])/relative,identity['sha256']))
 epoch.require_roster(roster)
 for path,digest in roster:require(epoch.digest_for(path,digest)==digest,'Current semantic pack witness differs')
 return {'current_pack_receipt_sha256':prepared['pack_receipt_sha256'],'members':len(roster),'explicit_operation_local_witness':True,'fresh_predevice_and_postoperation_required':True}

def for_prepared(directory,max_seconds=900):
 """Root-only payload boundary; authenticate fixed source37 pack metadata first."""
 from serial37_canonical_json_v3 import read_unique
 import c1_serve_controller_combined_v137 as c
 prepared=read_unique(Path(directory)/'prepared.json');require(Path(prepared['pack']).resolve()==c.PACK.resolve() and Path(prepared['pack_receipt']).resolve()==c.INTAKE.resolve(),'Exact original source37 pack/receipt path required before payload access');receipt=Path(prepared['pack_receipt']);raw=receipt.read_bytes();require(hashlib.sha256(raw).hexdigest()==prepared['pack_receipt_sha256'],'Current original pack receipt changed');value=read_unique(receipt);require(receipt.read_bytes()==raw,'Pack receipt changed while roster read');roster=[]
 for name,identity in value['RESULT']['files'].items():
  relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Invalid pack member path');roster.append((c.PACK/relative,identity['sha256']))
 return PackEpoch(roster,max_seconds)
