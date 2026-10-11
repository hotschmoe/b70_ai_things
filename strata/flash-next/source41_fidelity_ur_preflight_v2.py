"""Metadata-only necessary-condition refusal; never model/native qualification."""
import hashlib,json
from pathlib import Path
from audit_prefix30_lifecycle_v1 import audit_text
from serial37_canonical_json_v3 import read_unique
LIMIT=128<<20
def require(ok,msg):
 if not ok:raise ValueError(msg)
def consume(path,limit):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'Exact regular preflight artifact required')
 def sig():
  s=path.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 before=sig();require(before[2]<=limit,'Bounded metadata/log preflight required');raw=path.read_bytes();require(len(raw)==before[2]and sig()==before and path.read_bytes()==raw and sig()==before,'Preflight artifact changed');return raw
def preflight(root):
 root=Path(root).resolve();capture=root/'source41-original-fidelity';bindings={};values={}
 for name in('recipe.json','command.json','result.json'):
  path=capture/name;raw=consume(path,2<<20);bindings[name]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};values[name]=read_unique(path)
 recipe=values['recipe.json'];command=values['command.json'];result=values['result.json']
 require(recipe['engine_generation']==41 and type(recipe['engine_generation'])is int and recipe['materialized_mode']==1 and type(recipe['materialized_mode'])is int and recipe['observer_enabled']is True,'Exact source41 mode1 observer capture required')
 declared=recipe['native_env'].get('SYCL_UR_TRACE');sent=[command[i+1]for i,x in enumerate(command[:-1])if x=='-e'and command[i+1].startswith('SYCL_UR_TRACE=')];actual=[x for x in result['terminal']['Config']['Env']if x.startswith('SYCL_UR_TRACE=')]
 require(declared=='2'and sent==actual==['SYCL_UR_TRACE=2'],'Missing exact actual UR owning-allocation/free tracing: recipe/command/terminal must declare2')
 log=consume(capture/'engine.combined.log',LIMIT);bindings['engine.combined.log']={'path':str(capture/'engine.combined.log'),'bytes':len(log),'sha256':hashlib.sha256(log).hexdigest()};require(log.endswith(b'\n')and log.isascii(),'Complete ASCII preflight log required')
 for marker in(b'P30 allocation ',b'SFD allocation ',b'P30 release_begin ',b'P30 release_returned ',b'urUSMDeviceAlloc',b'urUSMFree'):require(marker in log,'Missing necessary actual lifetime marker '+marker.decode('ascii'))
 audit=audit_text(log.decode('ascii'),{0:(0,48)},True);require(audit['passed'],'Necessary actual P30/SFD owning-allocation/free audit failed: '+json.dumps(audit['errors'],ensure_ascii=True))
 return {'scope':'metadata-only necessary UR lifetime conditions; no actor/model/identity authority','artifacts':bindings,'audit':audit,'full_model_identity_qualified':False,'full_model_math_qualified':False,'actor_public_admission_reexecuted':False}
