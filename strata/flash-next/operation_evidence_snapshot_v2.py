"""Unintegrated process-local immutable evidence bytes/pure-result prototype.
Live health/time/model/page/lease/container guards are deliberately outside
semantic reuse. No saved snapshot/result admission constructor exists.
"""
import copy,hashlib,os,time,marshal,inspect,dis,builtins,types
from types import MappingProxyType
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from operation_pack_hash_witness_v2 import require,stat5
MAX_FILE_BYTES=128<<20;MAX_TOTAL_BYTES=512<<20;MAX_FILES=512
def immutable_state(value):
 if value is None or type(value)in (bool,int,float,str):canonical(value);return value
 if type(value)is tuple:return tuple(immutable_state(item) for item in value)
 raise ValueError('Mutable/nonprimitive global state cannot enter pure memo contract')

PURE_BUILTINS={'len','str','int','float','bool','bytes','dict','list','tuple','sum','min','max','abs','round','enumerate','zip','range','sorted','all','any','isinstance','type','ValueError','AssertionError'}
def instructions(code):
 rows=list(dis.get_instructions(code))
 for constant in code.co_consts:
  if isinstance(constant,types.CodeType):rows.extend(instructions(constant))
 return rows

def validator_state(validator,bindings):
 require(inspect.isfunction(validator),'Exact unbound Python validator required')
 require(not validator.__closure__ and not validator.__defaults__ and not validator.__kwdefaults__,'Closures/defaults are unsupported pure memo state')
 rows=instructions(validator.__code__);require(not any(r.opname in ('STORE_GLOBAL','DELETE_GLOBAL','STORE_DEREF','DELETE_DEREF','IMPORT_NAME','IMPORT_FROM') for r in rows),'State mutation/import is unsupported pure memo work');require(not any(r.opname in ('LOAD_ATTR','LOAD_METHOD') and str(r.argval).startswith('_') for r in rows),'Private/reflection attribute dependency is not pure input')
 names={row.argval for row in rows if row.opname=='LOAD_GLOBAL'};current={}
 for name in names:
  if name in validator.__globals__:current[name]=immutable_state(validator.__globals__[name])
  else:require(name in PURE_BUILTINS and name in validator.__builtins__ and validator.__builtins__[name] is getattr(builtins,name,None),'Unknown/shadowed/impure builtin dependency')
 require(type(bindings)is dict and set(bindings)==set(current),'Complete explicit immutable global binding required');require(canonical(bindings)==canonical(current),'Current global state changed from supplied complete binding');return current

class ImmutableGateInputs:
 __slots__=('_data',)
 def __init__(self,data):object.__setattr__(self,'_data',MappingProxyType(data))
 def bytes_for(self,path):
  path=str(Path(path).resolve());require(path in self._data,'Undeclared gate-specific evidence input');return self._data[path]
 def __setattr__(self,name,value):raise ValueError('Immutable gate input mutation refused')

class EvidenceEpoch:
 def __init__(self,roster,max_seconds=900):
  require(type(max_seconds)is int and 1<=max_seconds<=10800,'Bounded evidence operation required');self.owner=os.getpid();self.started=time.monotonic();self.max_seconds=max_seconds;self.phase='admission';self.roster={};self.buffers={};self.boundaries=[];self.memo={};self.active_inputs=[];self.in_progress=set();self.semantic_calls=0;self.semantic_executions=0
  for path,digest in roster:
   original=Path(path);require(original.is_file() and not original.is_symlink() and original.absolute()==original.resolve(),'Regular unaliased evidence path required');path=str(original.resolve());require(path not in self.roster,'Duplicate evidence path');require(digest is None or type(digest)is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Declared expected SHA or explicit unpinned-current input required');self.roster[path]=digest
  require(1<=len(self.roster)<=MAX_FILES,'Bounded complete evidence roster required');self.initial=self._fresh('entry',capture=True)
 def _owned(self):require(os.getpid()==self.owner and time.monotonic()-self.started<=self.max_seconds,'Evidence epoch process/lifetime changed')
 def _fresh(self,label,capture=False):
  self._owned();start=time.time();rows=[];total=0
  for path,expected in sorted(self.roster.items()):
   p=Path(path);require(not p.is_symlink(),'Evidence path became symlink');before=stat5(p);require(before[2]<=MAX_FILE_BYTES and total+before[2]<=MAX_TOTAL_BYTES,'Evidence byte cap exceeded before read');chunks=[];h=hashlib.sha256();count=0
   with p.open('rb') as handle:
    fd=os.fstat(handle.fileno());require([fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns]==before,'Evidence descriptor/path differs')
    while True:
     block=handle.read(min(8<<20,MAX_FILE_BYTES-count+1))
     if not block:break
     count+=len(block);require(count<=MAX_FILE_BYTES,'Evidence grew beyond byte bound');h.update(block)
     if capture:chunks.append(block)
    fd=os.fstat(handle.fileno())
   after=stat5(p);require(before==after==[fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns] and count==before[2],'Evidence mutated during complete byte read');digest=h.hexdigest();require(expected is None or digest==expected,'Evidence bytes differ declared expected SHA');row={'path':path,'sha256':digest,'bytes':count,'stat_before':before,'stat_after':after,'externally_expected_sha256':expected};rows.append(row);total+=count
   if capture:self.buffers[path]=b''.join(chunks)
  if hasattr(self,'initial'):require(canonical(rows)==canonical(self.initial),'Evidence bytes/roster/stat changed across full boundaries')
  self.boundaries.append({'label':label,'started_epoch':start,'finished_epoch':time.time(),'rows':rows,'total_bytes':total});return rows
 def require_roster(self,roster):
  self._owned();require(self.phase in ('admission','sealed'),'Closed evidence epoch');current={}
  for path,digest in roster:
   path=str(Path(path).resolve());require(path not in current,'Duplicate caller evidence roster');current[path]=digest
  require(canonical(current)==canonical(self.roster),'Current complete expected evidence roster changed');return True
 def bytes_for(self,path):
  self._owned();require(self.phase in ('admission','sealed'),'Closed evidence snapshot');path=str(Path(path).resolve());require(path in self.buffers and (not self.active_inputs or path in self.active_inputs[-1]),'Undeclared gate-specific evidence input');return self.buffers[path]
 def pure_once(self,gate_name,validator_source_path,validator_source_sha256,parameters,inputs,validator,*,current_roster,global_bindings):
  self._owned();self.require_roster(current_roster);require(self.phase in ('admission','sealed'),'Closed semantic epoch');require(type(gate_name)is str and gate_name,'Explicit pure gate name required');source=str(Path(validator_source_path).resolve());require(source in self.buffers and hashlib.sha256(self.buffers[source]).hexdigest()==validator_source_sha256,'Exact current validator source snapshot required');paths=[str(Path(p).resolve()) for p in inputs];require(paths and len(paths)==len(set(paths)) and set(paths)<=set(self.roster),'Exact distinct declared gate input subset required');require(hasattr(validator,'__code__'),'Explicit Python pure validator code required');state=validator_state(validator,global_bindings);fingerprint=hashlib.sha256(marshal.dumps(validator.__code__)).hexdigest();key=canonical({'validator_global_state':state,'validator_module':validator.__module__,'validator_qualname':validator.__qualname__,'validator_code_sha256':fingerprint,'gate':gate_name,'validator_source':source,'validator_sha256':validator_source_sha256,'parameters':parameters,'inputs':sorted(paths)});self.semantic_calls+=1
  require(key not in self.in_progress,'Recursive duplicate pure gate; declared DAG cycle refused')
  if key not in self.memo:
   # Future port receives this explicit immutable snapshot; no read/sha patch.
   self.in_progress.add(key);self.active_inputs.append(set(paths)|{source})
   try:view=ImmutableGateInputs({path:self.buffers[path] for path in set(paths)|{source}});result=validator(view,copy.deepcopy(parameters));canonical(result);self.memo[key]=copy.deepcopy(result);self.semantic_executions+=1
   finally:self.active_inputs.pop();self.in_progress.remove(key)
  return copy.deepcopy(self.memo[key])
 def seal_predevice(self):
  self._owned();require(self.phase=='admission','Duplicate evidence seal');self._fresh('predevice');self.phase='sealed'
 def finalize(self):
  self._owned();require(self.phase=='sealed','Evidence predevice seal required');self._fresh('postoperation');self.phase='closed'
  return {'schema':1,'owner_pid':self.owner,'boundaries':self.boundaries,'semantic_calls':self.semantic_calls,'semantic_executions':self.semantic_executions,'memoized_pure_keys':len(self.memo),'externally_unpinned_current_inputs':[p for p,d in self.roster.items() if d is None],'saved_digest_or_result_imported':False,'stat_only_validation':False,'between_complete_byte_boundaries_mutation_unobserved':True,'live_guards_reuse_allowed':False,'full_semantic_or_runtime_admission_proven':False,'actual_runtime_integration':False}
