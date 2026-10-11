"""Per-service loaded identity/incarnation guard; never auto-rebind stale cache."""
import hashlib,json
from live_cache_identity_namespace_v2 import namespace,require,digest

def small_hash(value):
 raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii');require(len(raw)<=32<<20,'Bounded loaded metadata snapshot required');return hashlib.sha256(raw).hexdigest()

def metadata(service):
 engine=service.engine;proc=engine.proc
 require(proc is not None and type(proc.pid)is int and proc.pid>0 and type(engine.gen)is int and engine.gen>0 and proc.poll()is None,'Actual live native incarnation required')
 tok=service.tok;template=service.template
 tokenizer={'tokens':tok.tokens,'types':tok.token_types,'special':tok.special_ids,'pre':tok.pre,'ranks':sorted((a,b,rank)for(a,b),rank in tok.ranks.items())}
 return {'native_pid':proc.pid,'native_generation':engine.gen,'artifact_metadata_sha256':small_hash(service.artifact_identity),'tokenizer_loaded_metadata_sha256':small_hash(tokenizer),'template_loaded_source_sha256':small_hash(template.source),'launch_configuration_sha256':small_hash([engine.spawn[0],engine.spawn[1],engine.spawn[2],engine.spawn[4]])}

class LoadedOwner:
 def __init__(self,service,owner):
  self.namespace=namespace(owner);self.service=service;self.engine=service.engine;self.proc=service.engine.proc;self.tokenizer=service.tok;self.template=service.template;self.observed=metadata(service)
 def current(self,service):
  require(service is self.service and service.engine is self.engine and service.engine.proc is self.proc and service.tok is self.tokenizer and service.template is self.template,'LIVE_LOADED_OWNER_REFUSED object replaced')
  require(metadata(service)==self.observed,'LIVE_LOADED_OWNER_REFUSED incarnation/model/tokenizer/template/source configuration changed')
  return {'loaded_namespace_sha256':digest(self.namespace),'native_incarnation':{'pid':self.observed['native_pid'],'generation':self.observed['native_generation']},'actual_loaded_metadata_snapshot':self.observed,'whole_weight_mutation_observed':False}
 def request_rebind(self,candidate):
  namespace(candidate)
  if self.proc.poll()is None:raise ValueError('LIVE_NAMESPACE_REBIND_REFUSED original native incarnation/cache still exists')
  raise ValueError('LIVE_NAMESPACE_REBIND_REFUSED fresh actual load/empty-cache admission required; no implicit owner update')
