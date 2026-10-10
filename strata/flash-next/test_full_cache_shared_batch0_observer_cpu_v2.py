import ast,copy,unittest
from full_cache_shared_batch0_api_trace_v2 import cfg_gate,adapted_source

class SerialObserver(unittest.TestCase):
 def fixture(self):return {'args':['--batch','0','--prompt-cache','3'],'parallel':1,'slot_save_path':'/results/sessions','env':{'STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1'}}
 def test_actual_adapter_source_session_hook_only(self):
  source=adapted_source();ast.parse(source);self.assertIn('session_observer(server,emit,local,counter,lock,active_calls)',source);self.assertIn('iterator=generate_original(self,ids,max_new,sampling,cancel,embeddings)',source);self.assertNotIn('batch0_identity',source);self.assertNotIn('_request_keys(sampling or {})',source)
 def test_serial_normal_PCL_scope_and_batch_capability_refused(self):
  cfg=self.fixture();cfg_gate(cfg)
  for key,value in [('STRATA_BATCH_FIDELITY_DIAG','1'),('STRATA_FULL_CACHE_OBSERVER38','1'),('STRATA_BATCH_PUBLIC_PREFIX','1'),('STRATA_VERIFY_EAGER','0')]:
   row=copy.deepcopy(cfg);row['env'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):cfg_gate(row)
  cfg['args'][1]='2'
  with self.assertRaises(ValueError):cfg_gate(cfg)
if __name__=='__main__':unittest.main()
