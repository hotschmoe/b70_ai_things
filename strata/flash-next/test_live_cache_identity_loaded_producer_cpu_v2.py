"""Actual guard execution with tiny own service, never native model observations."""
import copy,types,unittest,time
from unittest.mock import patch
from full_cache_live_identity_api_v2 import install,PROBE_FIELD
from live_cache_identity_namespace_v2 import KEYS
from run_live_cache_identity_probes_v2 import recollect_loaded
class Producer(unittest.TestCase):
 def fixture(self):
  events=[];calls=[]
  class Service:
   def prepare(self,*args,**kwargs):calls.append('original');return 'original'
  owner={k:'a'*64 for k in KEYS};nonce='b'*64;install(Service,owner,[],events.append,probe_nonce=nonce)
  svc=Service();svc.engine=types.SimpleNamespace(proc=types.SimpleNamespace(pid=17,poll=lambda:None),gen=1,spawn=['exe',[],'.','log',{},False,None],restart=lambda:self.fail('original restart ran'));svc.tok=types.SimpleNamespace(tokens=['word'],token_types=[1],special_ids={},pre='qwen35',ranks={});svc.template=types.SimpleNamespace(source='original');svc.artifact_identity={'original':'metadata'}
  return svc,events,calls,nonce
 def test_actual_all_mutations_restored_and_replay_after_refusal(self):
  svc,guards,calls,nonce=self.fixture();start=time.time();native=[{'kind':kind,'epoch':start-offset,'engine_pid':17,'engine_generation':1}for kind,offset in [('engine_begin',2),('engine_end',1)]];census={'native_pid':17,'native_generation':1,'native_begin_count':1,'native_terminal_count':1};rows=[]
  with patch('full_cache_live_identity_api_v2.watched_bytes',return_value=True):
   self.assertEqual(svc.prepare([],None,{},req={}), 'original')
   for component in (*KEYS,'authoritative_rebind','native_restart'):
    before=time.time();request={PROBE_FIELD:{'component':component,'nonce':nonce}}
    with self.assertRaises(ValueError)as failed:svc.prepare([],None,{},req=request)
    guard=guards[-1];rows.append({'component':component,'actual_request':request,'response_status':400,'response_hex':str(failed.exception).encode().hex(),'started_epoch':before,'finished_epoch':time.time(),'native_before':dict(census),'native_after':dict(census),'original_guard_rows':[guard],'actual_other_model_weights_loaded':False})
   self.assertEqual(svc.prepare([],None,{},req={}), 'original')
  proof=recollect_loaded(rows,native,guards);self.assertTrue(proof['all_four_actual_loaded_metadata_changes_refused']);self.assertEqual(calls,['original','original']);self.assertEqual(svc.tok.tokens,['word']);self.assertEqual(svc.template.source,'original')
  changed=copy.deepcopy(rows);changed[0]['original_guard_rows'][0]['loaded_metadata_mutation_attempted']=False
  with self.assertRaises(ValueError):recollect_loaded(changed,native,[changed[0]['original_guard_rows'][0],*guards[1:]])
 def test_original_restart_is_refused_before_owner_replacement(self):
  svc,events,calls,nonce=self.fixture()
  with patch('full_cache_live_identity_api_v2.watched_bytes',return_value=True):svc.prepare([],None,{},req={})
  with self.assertRaisesRegex(ValueError,'RE BIND'.replace(' ','')):svc.engine.restart()
  self.assertEqual(calls,['original']);self.assertEqual(svc.engine.proc.pid,17)
if __name__=='__main__':unittest.main()
