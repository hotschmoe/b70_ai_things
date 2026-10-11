import hashlib,tempfile,unittest
from pathlib import Path
from live_cache_identity_namespace_v1 import admit,namespace_probes,watched_bytes,KEYS,FIELD
from full_cache_live_identity_api_v1 import install,PROBE_FIELD

class Identity(unittest.TestCase):
 def owner(self):return {k:format(i+1,'064x')for i,k in enumerate(KEYS)}
 def test_all_four_foreign_claims_refuse_without_owner_mutation(self):
  owner=self.owner();before=dict(owner)
  for probe in namespace_probes(owner):
   with self.assertRaisesRegex(ValueError,probe['expected_refusal']):admit(owner,{FIELD:probe['requested_namespace']})
  self.assertEqual(owner,before);self.assertFalse(admit(owner,None)['native_cache_key_model_fingerprint_observed'])
 def test_exact_explicit_owner_and_missing_default_are_original(self):
  self.assertTrue(admit(self.owner(),{FIELD:self.owner()})['explicit_namespace_supplied']);self.assertFalse(admit(self.owner(),{})['explicit_namespace_supplied'])
 def test_unknown_partial_and_wrong_type_namespace_rejected(self):
  for value in (None,{},dict(self.owner(),extra='0'*64),[],1):
   with self.assertRaises(ValueError):admit(self.owner(),{FIELD:value})
 def test_changed_watched_current_bytes_refused_not_silently_rebound(self):
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'metadata';path.write_bytes(b'original');rows=[{'path':str(path),'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}];self.assertTrue(watched_bytes(rows));path.write_bytes(b'foreign!')
   with self.assertRaises(ValueError):watched_bytes(rows)
 def test_producer_shaped_prepare_does_not_render_or_touch_cache_on_refusal(self):
  import types
  calls=[];events=[]
  class Service:
   def prepare(self,*args,**kwargs):calls.append(kwargs);return 'original'
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'metadata';path.write_bytes(b'x');watched=[{'path':str(path),'bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}];saved=install(Service,self.owner(),watched,events.append,probe_nonce='n'*64);svc=Service();svc.engine=types.SimpleNamespace(proc=types.SimpleNamespace(pid=17,poll=lambda:None),gen=1,spawn=['exe',[],'.','log',{},False,None]);svc.tok=types.SimpleNamespace(tokens=['word'],token_types=[1],special_ids={},pre='qwen35',ranks={});svc.template=types.SimpleNamespace(source='original');svc.artifact_identity={'original':'metadata'}
   for probe in namespace_probes(self.owner()):
    with self.assertRaises(ValueError):svc.prepare([],None,{},req={FIELD:probe['requested_namespace']})
   self.assertEqual(calls,[]);self.assertTrue(all(e['original_prepare_invoked']is False and e['native_request_authorized']is False for e in events));self.assertEqual(svc.prepare([],None,{},req={}), 'original');self.assertEqual(len(calls),1)
   original_code=saved.__code__
   for component in KEYS:
    with self.assertRaisesRegex(ValueError,'LIVE_LOADED_OWNER_REFUSED'):svc.prepare([],None,{},req={PROBE_FIELD:{'component':component,'nonce':'n'*64}})
    self.assertEqual(svc.tok.tokens,['word']);self.assertEqual(svc.template.source,'original');self.assertEqual(svc.artifact_identity,{'original':'metadata'});self.assertIs(saved.__code__,original_code)
   self.assertEqual(len(calls),1);self.assertTrue(all(e['actual_model_weight_data_changed']is False for e in events if e['kind']=='live_identity_prepare_refused'))
 def test_loaded_owner_change_or_restart_cannot_auto_rebind(self):
  import types
  from live_cache_loaded_owner_v1 import LoadedOwner
  service=types.SimpleNamespace(engine=types.SimpleNamespace(proc=types.SimpleNamespace(pid=17,poll=lambda:None),gen=1,spawn=['exe',[],'.','log',{},False,None]),tok=types.SimpleNamespace(tokens=['word'],token_types=[1],special_ids={},pre='qwen35',ranks={}),template=types.SimpleNamespace(source='original'),artifact_identity={'original':'metadata'});owner=LoadedOwner(service,self.owner());self.assertFalse(owner.current(service)['whole_weight_mutation_observed'])
  with self.assertRaisesRegex(ValueError,'original native incarnation'):owner.request_rebind(self.owner())
  service.template.source='changed'
  with self.assertRaises(ValueError):owner.current(service)
  service.template.source='original';service.engine.gen=2
  with self.assertRaises(ValueError):owner.current(service)
  service.engine.gen=1;service.tok=types.SimpleNamespace(tokens=['word'],token_types=[1],special_ids={},pre='qwen35',ranks={})
  with self.assertRaises(ValueError):owner.current(service)

 def test_guarded_actual_restart_method_refuses_before_old_owner_close(self):
  import types
  calls=[];events=[]
  class Service:
   def prepare(self,*args,**kwargs):return 'original'
  install(Service,self.owner(),[],events.append)
  # Bounded watched metadata remains mandatory; patch only this CPU fixture's
  # watch boundary, never the owning metadata/current/native restart guard.
  from unittest.mock import patch
  svc=Service();svc.engine=types.SimpleNamespace(proc=types.SimpleNamespace(pid=17,poll=lambda:None),gen=1,spawn=['exe',[],'.','log',{},False,None],restart=lambda:calls.append('old-owner-close'))
  svc.tok=types.SimpleNamespace(tokens=['word'],token_types=[1],special_ids={},pre='qwen35',ranks={});svc.template=types.SimpleNamespace(source='original');svc.artifact_identity={'original':'metadata'}
  with patch('full_cache_live_identity_api_v1.watched_bytes',return_value=True):svc.prepare([],None,{},req={})
  with self.assertRaisesRegex(ValueError,'original native incarnation'):svc.engine.restart()
  self.assertEqual(calls,[]);self.assertEqual(svc.engine.proc.pid,17)
  svc.engine.proc.poll=lambda:0
  with self.assertRaisesRegex(ValueError,'fresh actual load/empty-cache'):svc.engine.restart()
  self.assertEqual(calls,[])
