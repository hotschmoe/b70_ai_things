"""Process-local executable-byte epoch prototype; no saved proof import.
Explicit consumers still run unchanged source/header/recipe/health semantics.
This module is not integrated into H48 or any frozen admission function.
"""
import hashlib,json,os,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique,unique_object,finite_constant
from operation_pack_hash_witness_v2 import require,stat5
TARGETS=('strata','native_expert_parity','conversation_snapshot_test','iq_multi_parity','native_grouped_parity','native_multi_parity','shared_expert_parity','verify_parity')
ROLES=tuple('SDK37:'+name for name in TARGETS)+('source37-upload-oracle',)

def expected(roster):
 rows={};paths=set()
 for role,path,digest in roster:
  path=str(Path(path).resolve());require(role in ROLES and role not in rows and path not in paths,'Exact unique SDK37/oracle role/path required');require(type(digest)is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Exact expected executable SHA required');rows[role]={'path':path,'sha256':digest};paths.add(path)
 require(set(rows)==set(ROLES),'Complete eight SDK37 ELF plus upload-oracle roster required');return rows

def metadata_bytes(path):
 path=Path(path).resolve();before=stat5(path);require(before[2]<=16<<20,'Receipt exceeds preregistered bound before any byte read')
 with path.open('rb') as handle:
  fd=os.fstat(handle.fileno());require([fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns]==before,'Receipt path/FD differs');raw=handle.read((16<<20)+1);fd=os.fstat(handle.fileno())
 require(len(raw)<=16<<20 and len(raw)==before[2] and stat5(path)==before==[fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns],'Receipt changed/grown during bounded byte read');return raw,before

def fresh_receipts(receipts):
 result=[]
 for path,digest in receipts:
  path=Path(path).resolve();raw,before=metadata_bytes(path);require(hashlib.sha256(raw).hexdigest()==digest,'Current expected receipt SHA differs');result.append({'path':str(path),'sha256':digest,'bytes':len(raw),'stat5':before})
 require(2<=len(result)<=8 and len({r['path'] for r in result})==len(result),'Bounded unique engine/oracle/plan receipt inputs required');return sorted(result,key=lambda r:r['path'])

def read_elf(path):
 path=Path(path).resolve();before=stat5(path);h=hashlib.sha256();count=0;prefix=b''
 with path.open('rb') as handle:
  fd=os.fstat(handle.fileno());require([fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns]==before,'Executable descriptor/path changed')
  while True:
   block=handle.read(8<<20)
   if not block:break
   if len(prefix)<4:prefix=(prefix+block)[:4]
   count+=len(block);h.update(block)
  fd=os.fstat(handle.fileno())
 after=stat5(path);require(before==after==[fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns] and count==before[2],'Executable changed during full byte scan');require(prefix==b'\x7fELF','Actual complete-read ELF magic required')
 return {'path':str(path),'sha256':h.hexdigest(),'bytes':count,'ELF_magic_hex':prefix.hex(),'stat_before':before,'stat_after':after}

class SDK37Epoch:
 def __init__(self,roster,receipt_bindings,max_seconds=900):
  require(type(max_seconds)is int and 1<=max_seconds<=10800,'Bounded SDK epoch required');self.owner=os.getpid();self.started=time.monotonic();self.max_seconds=max_seconds;self.phase='admission';self.expected=expected(roster);self.receipts=list(receipt_bindings);self.current_receipts=fresh_receipts(self.receipts);self.boundaries=[];self.digest_calls=0;self.magic_calls=0;self.initial=self._fresh('entry')
 def _owned(self):require(os.getpid()==self.owner and time.monotonic()-self.started<=self.max_seconds,'SDK epoch process ownership/lifetime differs')
 def _fresh(self,label):
  self._owned();start=time.time();require(canonical(fresh_receipts(self.receipts))==canonical(self.current_receipts),'SDK/oracle receipt inputs changed');rows=[]
  for role,item in sorted(self.expected.items()):
   row=read_elf(item['path']);require(row['sha256']==item['sha256'],'Current executable bytes differ expected receipt');row['role']=role;rows.append(row)
  if hasattr(self,'initial'):require(canonical(rows)==canonical(self.initial),'SDK/oracle bytes or stat5 changed across fresh boundaries')
  require(canonical(fresh_receipts(self.receipts))==canonical(self.current_receipts),'SDK/oracle receipts changed during scan');self.boundaries.append({'label':label,'started_epoch':start,'finished_epoch':time.time(),'receipt_inputs':self.current_receipts,'rows':rows});return rows
 def require_current_roster(self,roster,receipt_bindings):
  self._owned();require(self.phase in ('admission','sealed'),'Closed SDK epoch refused');require(canonical(expected(roster))==canonical(self.expected) and canonical(fresh_receipts(receipt_bindings))==canonical(self.current_receipts),'Current semantic consumer expected roster/receipt differs');return True
 def digest_for(self,role,path,digest):
  self._owned();require(self.phase in ('admission','sealed'),'Closed SDK epoch refused');require(role in self.expected and self.expected[role]=={'path':str(Path(path).resolve()),'sha256':digest},'Unknown/stale executable semantic consumer');self.digest_calls+=1;return digest
 def ELF_magic(self,role,path,digest):
  self.digest_for(role,path,digest);self.magic_calls+=1;return bytes.fromhex(next(r['ELF_magic_hex'] for r in self.initial if r['role']==role))
 def seal_predevice(self):
  self._owned();require(self.phase=='admission','Duplicate SDK seal');self._fresh('predevice');self.phase='sealed';return self.boundaries[-1]
 def finalize(self):
  self._owned();require(self.phase=='sealed','Predevice SDK seal required');self._fresh('postoperation');self.phase='closed'
  return {'schema':1,'owner_pid':self.owner,'unique_executables':9,'digest_calls':self.digest_calls,'ELF_magic_calls':self.magic_calls,'boundaries':self.boundaries,'stat_only_validation':False,'saved_digest_imported':False,'between_byte_boundaries_mutation_unobserved':True,'full_source_header_recipe_health_or_runtime_admission_proven':False,'source37_ELF_marker_scan_invented':False,'actual_runtime_integration':False}

def metadata_roster(prepared_path):
 """Root-only future setup: metadata reads only; caller still validates baseline."""
 import c1_serve_controller_combined_v137 as c
 prepared_path=Path(prepared_path).resolve();prepared_raw,_=metadata_bytes(prepared_path);prepared=json.loads(prepared_raw,object_pairs_hook=unique_object,parse_constant=finite_constant);engine_receipt=Path(prepared['engine_receipt']);oracle_receipt=Path(prepared['upload_lifecycle']['oracle']);engine_raw,_=metadata_bytes(engine_receipt);oracle_raw,_=metadata_bytes(oracle_receipt);plan_raw,_=metadata_bytes(c.COMBINED_PLAN)
 require(hashlib.sha256(engine_raw).hexdigest()==prepared['engine_receipt_sha256'] and hashlib.sha256(oracle_raw).hexdigest()==prepared['upload_lifecycle']['oracle_sha256'],'Expected actual SDK/oracle receipt changed');engine=engine_receipt.parent;build=json.loads(engine_raw,object_pairs_hook=unique_object,parse_constant=finite_constant);oracle=json.loads(oracle_raw,object_pairs_hook=unique_object,parse_constant=finite_constant);plan=json.loads(plan_raw,object_pairs_hook=unique_object,parse_constant=finite_constant)
 require(hashlib.sha256(plan_raw).hexdigest()==c.COMBINED_PLAN_SHA==build['plan_sha256'] and plan['build_targets']==list(TARGETS),'Exact source37 combined recipe/targets required');require(build['build_rc']==0 and oracle['passed']is True and oracle['engine_receipt_sha256']==prepared['engine_receipt_sha256'],'Actual successful associated SDK/oracle metadata required')
 roster=[('SDK37:'+name,engine/'build'/name,build['binary_sha256'][str(engine/'build'/name)]) for name in TARGETS];roster.append(('source37-upload-oracle',oracle_receipt.parent/'source-upload-oracle',oracle['binary_sha256']))
 receipts=[(prepared_path,hashlib.sha256(prepared_raw).hexdigest()),(engine_receipt,prepared['engine_receipt_sha256']),(oracle_receipt,prepared['upload_lifecycle']['oracle_sha256']),(c.COMBINED_PLAN,c.COMBINED_PLAN_SHA)]
 return roster,receipts
