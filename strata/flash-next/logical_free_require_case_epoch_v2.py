"""Immutable success-token API for consumers that never expose the parsed ledger."""
from pathlib import Path
import hashlib
from logical_free_immutable_epoch_v2 import LogicalEpoch,require,canonical
HERE=Path(__file__).resolve().parent
SOURCES=('logical_free_require_case_epoch_v2.py','original_upload_logical_requirement_v2.py','logical_free_method_binding_v2.py')
PROTECTED=('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__')
class FrozenDispatch(type):
 def __setattr__(cls,name,value):
  if name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__') or name=='_source_method_codes':raise ValueError('Class method shadow refused '+name)
  return super().__setattr__(name,value)
 def __delattr__(cls,name):
  if name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__') or name=='_source_method_codes':raise ValueError('Class method removal refused '+name)
  return super().__delattr__(name)
 def __getattribute__(cls,name):
  value=super().__getattribute__(name)
  if name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__') and name!='_source_method_codes':
   data=super().__getattribute__('__dict__');known=data.get('_source_method_codes')
   if known is not None and name in known and (not hasattr(value,'__code__') or value.__code__!=known[name]):raise ValueError('Class method code shadow refused '+name)
  return value
class RequireLogicalEpoch(LogicalEpoch,metaclass=FrozenDispatch):
 def __setattr__(self,name,value):
  if name in ('_guard_source_codes','_known_method_cache'):raise ValueError('Private implementation cache injection refused')
  if name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__'):
   object.__setattr__(self,'failed',True);object.__setattr__(self,'binding_failed',True);raise ValueError('Instance method shadow refused '+name)
  object.__setattr__(self,name,value)
 def __getattribute__(self,name):
  if name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__') and name in object.__getattribute__(self,'__dict__'):
   object.__setattr__(self,'failed',True);object.__setattr__(self,'binding_failed',True);raise ValueError('Instance method dictionary shadow refused '+name)
  return object.__getattribute__(self,name)
 def __init__(self,roster,max_seconds=900):
  paths={Path(p).resolve()for p,d in roster};require(all(HERE/name in paths for name in SOURCES),'Explicit compact consumer/source closure required');super().__init__(roster,max_seconds);self.success_tokens={};self.snapshot_sha={row['path']:row['sha256']for row in self.evidence.initial};self.require_calls=0;self.first_result_isolation_copies=0;self.binding_failed=False;object.__setattr__(self,'_known_method_cache',None)
 def implementation_binding(self):
  try:
   import logical_free_method_binding_v2 as guard
   import types,sys
   raw=self.evidence.bytes_for(HERE/'logical_free_method_binding_v2.py');cache=vars(self).get('_guard_source_codes')
   if cache is None:
    root=compile(raw,str(HERE/'logical_free_method_binding_v2.py'),'exec',dont_inherit=True,optimize=0);stack=[root];codes={}
    while stack:
     code=stack.pop();codes[code.co_qualname]=code;stack.extend(v for v in code.co_consts if isinstance(v,types.CodeType))
    from types import MappingProxyType
    object.__setattr__(self,'_guard_source_codes',MappingProxyType(codes));cache=vars(self)['_guard_source_codes']
   codes=cache
   for name in ('binding','bind_function','require','code_map','compiled'):
    fn=getattr(guard,name)
    if type(fn)is not types.FunctionType or fn.__globals__ is not vars(guard) or fn.__module__!=guard.__name__ or fn.__qualname__!=name or fn.__code__!=codes[name]:raise ValueError('Known binding dispatcher differs '+name)
   return guard.binding(self)
  except BaseException as exc:
   self.binding_failed=True;self.failed=True;self.errors.append({'exception_type':type(exc).__name__,'error':str(exc),'method_binding_failure':True});raise
 def require_case(self,log,logical,expected_owner_count,*,current_roster):
  RequireLogicalEpoch.implementation_binding(self);LogicalEpoch.owned(self,current_roster);require(not self.failed,'Failed parser epoch cannot reuse success token');require(type(expected_owner_count)is int and expected_owner_count>=0,'Exact typed expected owner count');log=Path(log).resolve();logical=Path(logical).resolve();base=dict(self.base_roster);require(log in base and logical in base and log!=logical,'Declared distinct compact case inputs required');key=canonical({'execution_context':self.context,'runtime':self.runtime_key,'source_bytes':{name:hashlib.sha256(self.evidence.bytes_for(HERE/name)).hexdigest()for name in SOURCES},'log_path':str(log),'log_sha256':self.snapshot_sha[str(log)],'logical_path':str(logical),'logical_sha256':self.snapshot_sha[str(logical)],'require_owners':True,'expected_owner_count':expected_owner_count});self.require_calls+=1
  if key not in self.success_tokens:
   full=LogicalEpoch.collect(self,log,logical,expected_owner_count,current_roster=current_roster);self.worker_provenance(log,logical,expected_owner_count);require(full['original_upload_logical_subset_passed']is True,'Complete original logical subset must pass');self.first_result_isolation_copies+=1;self.success_tokens[key]=True
  # There is no shared mutable object to expose. Original complete-result
  # collect() API continues returning independent deep copies, unchanged.
  RequireLogicalEpoch.worker_provenance(self,log,logical,expected_owner_count)
  return True
 def worker_provenance(self,log,logical,expected_owner_count):
  import base64,os,sys
  from logical_free_immutable_epoch_v2 import canonical,digest,WORKER
  require(hashlib.sha256(self.evidence.bytes_for(log)).hexdigest()==dict(self.base_roster)[Path(log)] and hashlib.sha256(self.evidence.bytes_for(logical)).hexdigest()==dict(self.base_roster)[Path(logical)],'Worker operand bytes differ declared expected snapshot');request={'mode':'parse','parser_source_base64':base64.b64encode(self.source_raw).decode('ascii'),'log_base64':base64.b64encode(self.evidence.bytes_for(log)).decode('ascii'),'logical_base64':base64.b64encode(self.evidence.bytes_for(logical)).decode('ascii'),'require_owners':True,'expected_owner_count':expected_owner_count};packet=digest((canonical(request)+'\n').encode('ascii'));rows=[r for r in self.commands if r.get('input_sha256')==packet];require(rows and self.executions>0,'Genuine worker execution/packet provenance missing')
  for row in rows:
   require(row['worker_passed']is True and type(row['return_code'])is int and row['return_code']==0 and row['parse_counts']=={'positive':1,'negative':3} and all(type(x)is int for x in row['parse_counts'].values()),'Genuine worker result/counters required');require(row['command']==[str(Path(sys.executable).resolve()),'-I','-B','-S',str(WORKER)] and row['worker_identity']['pid']==row['child_pid'] and row['worker_identity']['parent_pid']==os.getpid() and row['worker_retirement']['worker_start_ticks']==row['worker_identity']['start_ticks'],'Genuine worker exact command/PID/start required');require(row['worker_retirement']['process_terminal']is True and row['worker_retirement']['owned_session_empty']is True and row['stdout_stderr_regular_sinks_closed']is True,'Actual worker retirement required')
 def finalize(self,*,current_roster):
  RequireLogicalEpoch.implementation_binding(self);proof=LogicalEpoch.finalize(self,current_roster=current_roster);proof.update(compact_requirement_calls=self.require_calls,first_full_result_isolation_copies=self.first_result_isolation_copies,compact_success_tokens=len(self.success_tokens),large_cached_result_exposed_by_requirement=False);return proof

# Code objects are immutable; protect expected dispatch from ordinary class edits.
from types import MappingProxyType
type.__setattr__(RequireLogicalEpoch,'_source_method_codes',MappingProxyType({name:getattr(RequireLogicalEpoch,name).__code__ for name in ('collect','owned','require_case','worker_provenance','finalize','seal_predevice','implementation_binding','__getattribute__','__setattr__')}))
