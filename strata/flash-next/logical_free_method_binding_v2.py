"""Bind declared known Python implementations, never arbitrary instance callbacks."""
import ast,hashlib,json,marshal,types
from pathlib import Path

def require(ok,message):
 if not ok:raise ValueError(message)
def code_map(code):
 result={code.co_qualname:code}
 for value in code.co_consts:
  if isinstance(value,types.CodeType):result.update(code_map(value))
 return result

def compiled(raw,path,epoch):
 from types import MappingProxyType
 cache=vars(epoch).get('_known_method_cache')
 if cache is None:cache=MappingProxyType({})
 key=(str(path),hashlib.sha256(raw).hexdigest())
 if key not in cache:
  codes=MappingProxyType(code_map(compile(raw,str(path),'exec',dont_inherit=True,optimize=0)));tree=ast.parse(raw);defaults={}
  def walk(nodes,prefix=''):
   for node in nodes:
    if isinstance(node,ast.ClassDef):walk(node.body,prefix+node.name+'.')
    elif isinstance(node,ast.FunctionDef):
     try:defaults[prefix+node.name]=(tuple(ast.literal_eval(v)for v in node.args.defaults)or None,{arg.arg:ast.literal_eval(value)for arg,value in zip(node.args.kwonlyargs,node.args.kw_defaults)if value is not None}or None)
     except ValueError:defaults[prefix+node.name]=None
  walk(tree.body);new=dict(cache);new[key]=(codes,MappingProxyType(defaults));cache=MappingProxyType(new);object.__setattr__(epoch,'_known_method_cache',cache)
 return cache[key]

def bind_function(function,module_name,qualname,epoch):
 evidence=epoch.evidence
 import sys
 require(function.__globals__ is vars(sys.modules[module_name]),'Declared function globals namespace differs '+qualname)
 require(type(function)is types.FunctionType and function.__module__==module_name and function.__qualname__==qualname and not function.__closure__,'Unknown method/function implementation');path=Path(function.__code__.co_filename).resolve();raw=evidence.bytes_for(path);expected,defaults_map=compiled(raw,path,epoch);require(qualname in expected and function.__code__==expected[qualname],'Declared method/class/global code differs '+qualname)
 require(defaults_map[qualname]is not None,'Unsupported bound function defaults '+qualname);defaults,kw=defaults_map[qualname]
 require(json.dumps(function.__defaults__,allow_nan=False)==json.dumps(defaults,allow_nan=False) and json.dumps(function.__kwdefaults__,sort_keys=True,allow_nan=False)==json.dumps(kw,sort_keys=True,allow_nan=False),'Declared function default state differs '+qualname)

def binding(epoch):
 import logical_free_require_case_epoch_v2 as compact
 import logical_free_immutable_epoch_v2 as base
 import operation_evidence_snapshot_v2 as ev
 import operation_pack_hash_witness_v2 as pack
 import serial37_canonical_json_v3 as canonical
 require(type(epoch)is compact.RequireLogicalEpoch and type(epoch.evidence)is ev.EvidenceEpoch,'Exact known epoch/evidence class required')
 classes=[(compact.RequireLogicalEpoch,('require_case','finalize','implementation_binding','worker_provenance')),(base.LogicalEpoch,('owned','collect','seal_predevice','finalize')),(ev.EvidenceEpoch,('_owned','_fresh','require_roster','bytes_for','seal_predevice','finalize'))]
 for cls,names in classes:
  instance=epoch.evidence if cls is ev.EvidenceEpoch else epoch
  for name in names:
   require(name not in vars(instance),'Instance method shadow refused '+name);fn=cls.__dict__[name];bind_function(fn,cls.__module__,cls.__name__+'.'+name,epoch)
 for module,names in [(base,('fixed_source','context','invoke','digest')),(pack,('require','stat5')),(canonical,('canonical','unique_object','finite_constant'))]:
  for name in names:bind_function(getattr(module,name),module.__name__,name,epoch)
 for module in (base,compact):
  bind_function(module.require,base.__name__,'require',epoch);bind_function(module.canonical,canonical.__name__,'canonical',epoch)
 require(compact.LogicalEpoch is base.LogicalEpoch,'Compact base-class alias differs')
 import original_upload_logical_requirement_v2 as requirement
 require(requirement.RequireLogicalEpoch is compact.RequireLogicalEpoch,'Original requirement class alias differs');bind_function(requirement.require,base.__name__,'require',epoch)
 import copy,json as json_module,sys as sys_module,pathlib,logical_free_worker_owned_v2 as owned
 for module in (base,compact):require(module.Path is pathlib.Path,'Path alias differs')
 require(base.copy is copy and base.json is json_module and base.sys is sys_module,'Base dispatch module aliases differ')
 bind_function(owned.execute,owned.__name__,'execute',epoch)
 # These are mutable dispatch symbols: compare to their preregistered source
 # qualnames rather than accepting whatever genuine function is substituted.
 for module,name in ((copy,'deepcopy'),(json_module,'loads')):
  path=Path(module.__file__).resolve();raw=path.read_bytes();fn=getattr(module,name);code,_=compiled(raw,path,epoch);require(type(fn)is types.FunctionType and fn.__module__==module.__name__ and fn.__qualname__==name and fn.__globals__ is vars(module) and fn.__code__==code[name],'Known stdlib dispatch function differs '+name)
 require(base.LogicalEpoch.collect is type(epoch).collect and base.LogicalEpoch.owned is type(epoch).owned,'Known inherited compact methods required')
 return True
