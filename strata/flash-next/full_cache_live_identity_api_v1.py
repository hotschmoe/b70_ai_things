"""ROOT container entry: new explicit prepare guard, unchanged SDK source."""
import hashlib,json,os,time
from pathlib import Path
import full_cache_shared_api_trace_v8 as original
from live_cache_identity_namespace_v1 import admit,namespace,watched_bytes,digest,FIELD
PROBE_FIELD='strata_loaded_owner_probe'

def mutation(service,saved,component):
 """Diagnostic-only attempted loaded owner update; always restore exact object."""
 if component=='model_sha256':
  original=service.artifact_identity;service.artifact_identity=dict(original,source_revision='0'*40,artifact_identity_sha256='0'*64)
  return lambda:setattr(service,'artifact_identity',original)
 if component=='tokenizer_sha256':
  original=service.tok.tokens[0];service.tok.tokens[0]=original+' LIVE_FOREIGN_OWNER'
  return lambda:service.tok.tokens.__setitem__(0,original)
 if component=='template_sha256':
  original=service.template.source;service.template.source=original+' LIVE_FOREIGN_OWNER'
  return lambda:setattr(service.template,'source',original)
 if component=='source_sha256':
  original=saved.__code__
  constants=tuple(value+' LIVE_FOREIGN_SOURCE'if type(value)is str else value for value in original.co_consts)
  saved.__code__=original.replace(co_consts=constants)
  return lambda:setattr(saved,'__code__',original)
 raise ValueError('Exact preregistered loaded component required')

def install(service,owner,watched,emit,expected_artifact_sha256=None,probe_nonce=None):
 import threading,weakref
 from live_cache_loaded_owner_v1 import LoadedOwner
 loaded=namespace(owner);saved=service.prepare;expected_code=saved.__code__;owners=weakref.WeakKeyDictionary();owner_lock=threading.RLock()
 def _guarded(self,messages,tools,kwargs,max_new=None,force=None,req=None):
  before=time.time();requested=req.get(FIELD)if type(req)is dict else None
  probe=req.get(PROBE_FIELD)if type(req)is dict else None;restore=None
  try:
   if probe is not None:
    if type(probe)is not dict or set(probe)!={'component','nonce'}or probe_nonce is None or probe['nonce']!=probe_nonce or self not in owners:raise ValueError('Diagnostic owner probe requires exact primed actor nonce')
    if probe['component']=='authoritative_rebind':
     candidate=dict(loaded);candidate['model_sha256']=('0'if candidate['model_sha256'][0]!='0'else'1')+candidate['model_sha256'][1:];owners[self].request_rebind(candidate)
    if probe['component']=='native_restart':
     self.engine.restart()
     raise ValueError('Unadmitted original restart returned; refuse original prepare')
    restore=mutation(self,saved,probe['component'])
   binding=admit(loaded,req);watched_bytes(watched)
   if saved.__code__ is not expected_code:raise ValueError('LIVE_LOADED_OWNER_REFUSED original source function replaced')
   if expected_artifact_sha256 is not None:
    if type(self.artifact_identity)is not dict or self.artifact_identity.get('artifact_identity_sha256')!=expected_artifact_sha256:raise ValueError('LIVE_LOADED_OWNER_REFUSED original validated load artifact differs')
   with owner_lock:
    if self not in owners:
     owners[self]=LoadedOwner(self,loaded)
     restart=getattr(self.engine,'restart',None)
     if callable(restart):
      current_service=self
      def guarded_restart(*args,**kwargs):
       # Refuse before original restart.close() destroys the primed native.
       # A fresh actor/full load/zero-cache full49 proof is required instead.
       with owner_lock:
        owners[current_service].request_rebind(loaded)
       raise ValueError('Unreachable unadmitted native restart')
      self.engine.restart=guarded_restart
    actual_owner=owners[self].current(self)
  except BaseException as error:
   emit({'kind':'live_identity_prepare_refused','started_epoch':before,'finished_epoch':time.time(),'loaded_namespace_sha256':digest(loaded),'requested_namespace':requested,'loaded_owner_probe':probe,'loaded_metadata_mutation_attempted':restore is not None,'actual_model_weight_data_changed':False,'error':type(error).__name__+': '+str(error),'original_prepare_invoked':False,'native_request_authorized':False,'loaded_owner_replaced':False})
   raise
  finally:
   if restore is not None:restore()
  if probe is not None:raise ValueError('Diagnostic loaded mutation unexpectedly accepted; original prepare still refused')
  emit({'kind':'live_identity_prepare_admitted','started_epoch':before,'finished_epoch':time.time(),'namespace_binding':binding,'actual_loaded_owner_binding':actual_owner,'original_prepare_invoked':True,'loaded_owner_replaced':False})
  return saved(self,messages,tools,kwargs,max_new=max_new,force=force,req=req)
 def guarded(self,*args,**kwargs):
  # Serialize attempted metadata updates, original prepare and exact restore.
  # No concurrent prepare can observe a diagnostic temporary owner mutation.
  with owner_lock:return _guarded(self,*args,**kwargs)
 service.prepare=guarded
 return saved

def main():
 import sys,threading
 sys.path[:0]=['/src','/src/tools'];from serve import server
 packet=Path('/results/live-identity-owner.json');raw=packet.read_bytes();owner=json.loads(raw);namespace(owner['namespace']);watched_bytes(owner['watched_files']);lock=threading.Lock();path=Path('/results/live-identity-guard.jsonl')
 def emit(row):
  row.update(owner_packet_sha256=hashlib.sha256(raw).hexdigest(),frontend_pid=os.getpid(),enforcement_scope='NEW_frontend_before_original_prepare_not_native_model_key')
  with lock:
   with path.open('a',encoding='ascii')as out:out.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n');out.flush()
 expected=owner['loaded_artifact_identity_sha256']
 if type(expected)is not str or len(expected)!=64:raise ValueError('Exact authoritative original load artifact digest required')
 install(server.Service,owner['namespace'],owner['watched_files'],emit,expected,hashlib.sha256(raw).hexdigest());original.main()
if __name__=='__main__':main()
