"""Actual-shaped source census and refusal re-reading, never GPU proof."""
import copy,unittest
from run_live_cache_identity_probes_v1 import census_at,recollect_loaded
from live_cache_identity_namespace_v1 import KEYS
from full_cache_live_identity_api_v1 import PROBE_FIELD
class Controls(unittest.TestCase):
 def fixture(self):
  events=[{'kind':'engine_begin','epoch':1.,'engine_pid':23,'engine_generation':1},{'kind':'engine_end','epoch':2.,'engine_pid':23,'engine_generation':1}];census=census_at(events,3);guards=[];rows=[]
  for index,component in enumerate((*KEYS,'authoritative_rebind','native_restart')):
   started=3+index*2;expected='LIVE_NAMESPACE_REBIND_REFUSED'if component in('authoritative_rebind','native_restart')else'LIVE_LOADED_OWNER_REFUSED';probe={'component':component,'nonce':'a'*64};guard={'kind':'live_identity_prepare_refused','started_epoch':started+.1,'finished_epoch':started+.2,'original_prepare_invoked':False,'native_request_authorized':False,'loaded_owner_replaced':False,'actual_model_weight_data_changed':False,'loaded_owner_probe':probe};guards.append(guard);rows.append({'component':component,'response_hex':expected.encode('ascii').hex(),'response_status':400,'actual_other_model_weights_loaded':False,'original_guard_rows':[guard],'actual_request':{PROBE_FIELD:probe},'native_before':census,'native_after':census,'started_epoch':started,'finished_epoch':started+.3})
  return rows,events,guards
 def test_complete_loaded_metadata_refusal_roster(self):
  rows,events,guards=self.fixture();value=recollect_loaded(rows,events,guards);self.assertTrue(value['live_authoritative_rebind_refused']);self.assertFalse(value['actual_other_model_weights_loaded']);self.assertFalse(value['new_loaded_incarnation_qualified'])
 def test_saved_counter_cannot_borrow_no_native_activity(self):
  rows,events,guards=self.fixture();rows=copy.deepcopy(rows);rows[0]['native_before']['native_begin_count']=9;rows[0]['native_after']['native_begin_count']=9;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
 def test_native_request_during_negative_and_wrong_guard_rejected(self):
  rows,events,guards=self.fixture();bad=events+[{'kind':'engine_begin','epoch':3.15,'engine_pid':23,'engine_generation':1}];self.assertRaises(ValueError,recollect_loaded,rows,bad,guards)
  rows=copy.deepcopy(rows);rows[0]['original_guard_rows'][0]['original_prepare_invoked']=True;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
 def test_truncated_roster_and_foreign_epoch_refused(self):
  rows,events,guards=self.fixture();self.assertRaises(ValueError,recollect_loaded,rows[:-1],events,guards);rows[0]['started_epoch']=4;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
if __name__=='__main__':unittest.main()
