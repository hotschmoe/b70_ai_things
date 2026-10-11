"""Actual-shaped source census and refusal re-reading, never GPU proof."""
import copy,unittest
from run_live_cache_identity_probes_v2 import census_at,recollect_loaded
from live_cache_identity_namespace_v2 import KEYS
from full_cache_live_identity_api_v2 import PROBE_FIELD
class Controls(unittest.TestCase):
 def fixture(self):
  events=[{'kind':'engine_begin','epoch':1.,'engine_pid':23,'engine_generation':1},{'kind':'engine_end','epoch':2.,'engine_pid':23,'engine_generation':1}];census=census_at(events,3);guards=[];rows=[]
  for index,component in enumerate((*KEYS,'authoritative_rebind','native_restart')):
   started=3+index*2;expected='LIVE_NAMESPACE_REBIND_REFUSED'if component in('authoritative_rebind','native_restart')else'LIVE_LOADED_OWNER_REFUSED';probe={'component':component,'nonce':'a'*64};guard={'kind':'live_identity_prepare_refused','started_epoch':started+.1,'finished_epoch':started+.2,'original_prepare_invoked':False,'native_request_authorized':False,'loaded_owner_replaced':False,'actual_model_weight_data_changed':False,'loaded_owner_probe':probe};guards.append(guard);rows.append({'component':component,'response_hex':expected.encode('ascii').hex(),'response_status':400,'actual_other_model_weights_loaded':False,'original_guard_rows':[guard],'actual_request':{PROBE_FIELD:probe},'native_before':census,'native_after':census,'started_epoch':started,'finished_epoch':started+.3})
  metadata={'native_pid':23,'native_generation':1,'artifact_metadata_sha256':'a'*64,'tokenizer_loaded_metadata_sha256':'a'*64,'template_loaded_source_sha256':'a'*64,'launch_configuration_sha256':'a'*64};owner={'loaded_namespace_sha256':'c'*64,'native_incarnation':{'pid':23,'generation':1},'actual_loaded_metadata_snapshot':metadata,'whole_weight_mutation_observed':False}
  fields={'model_sha256':'artifact_metadata_sha256','tokenizer_sha256':'tokenizer_loaded_metadata_sha256','template_sha256':'template_loaded_source_sha256'}
  for guard in guards:
   component=guard['loaded_owner_probe']['component'];attempted=component in KEYS;snapshot={'metadata':dict(metadata),'prepare_code_sha256':'a'*64}if attempted else None
   if component in fields:snapshot['metadata'][fields[component]]='b'*64
   if component=='source_sha256':snapshot['prepare_code_sha256']='b'*64
   expected='LIVE_NAMESPACE_REBIND_REFUSED'if not attempted else'LIVE_LOADED_OWNER_REFUSED'
   guard.update(loaded_metadata_mutation_attempted=attempted,loaded_metadata_restored=True,actual_loaded_owner_before_binding=copy.deepcopy(owner),restored_actual_loaded_owner_binding=copy.deepcopy(owner),prepare_code_sha256_before='a'*64,prepare_code_sha256_restored='a'*64,attempted_loaded_metadata_snapshot=snapshot,error='ValueError: '+expected)
  guards[:0]=[{'kind':'live_identity_prepare_admitted','started_epoch':2.1,'finished_epoch':2.2,'actual_loaded_owner_binding':copy.deepcopy(owner),'prepare_code_sha256':'a'*64,'original_prepare_invoked':True,'loaded_owner_replaced':False}]
  guards.append(dict(guards[0],started_epoch=20.,finished_epoch=20.1))
  return rows,events,guards
 def test_complete_loaded_metadata_refusal_roster(self):
  rows,events,guards=self.fixture();value=recollect_loaded(rows,events,guards);self.assertTrue(value['live_authoritative_rebind_refused']);self.assertFalse(value['actual_other_model_weights_loaded']);self.assertFalse(value['new_loaded_incarnation_qualified'])
 def test_false_missing_mutation_or_restore_never_claims_observation(self):
  for field,value in [('loaded_metadata_mutation_attempted',False),('loaded_metadata_mutation_attempted',None),('loaded_metadata_restored',False)]:
   rows,events,guards=self.fixture();rows[0]['original_guard_rows'][0][field]=value
   with self.assertRaises(ValueError):recollect_loaded(rows,events,guards)
 def test_unmutated_attempt_foreign_restore_or_missing_repeat_refused(self):
  rows,events,guards=self.fixture();event=rows[0]['original_guard_rows'][0];event['attempted_loaded_metadata_snapshot']['metadata']=dict(event['actual_loaded_owner_before_binding']['actual_loaded_metadata_snapshot']);self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
  rows,events,guards=self.fixture();rows[0]['original_guard_rows'][0]['restored_actual_loaded_owner_binding']={};self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
  rows,events,guards=self.fixture();self.assertRaises(ValueError,recollect_loaded,rows,events,guards[:-1])
 def test_saved_counter_cannot_borrow_no_native_activity(self):
  rows,events,guards=self.fixture();rows=copy.deepcopy(rows);rows[0]['native_before']['native_begin_count']=9;rows[0]['native_after']['native_begin_count']=9;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
 def test_native_request_during_negative_and_wrong_guard_rejected(self):
  rows,events,guards=self.fixture();bad=events+[{'kind':'engine_begin','epoch':3.15,'engine_pid':23,'engine_generation':1}];self.assertRaises(ValueError,recollect_loaded,rows,bad,guards)
  rows=copy.deepcopy(rows);rows[0]['original_guard_rows'][0]['original_prepare_invoked']=True;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
 def test_truncated_roster_and_foreign_epoch_refused(self):
  rows,events,guards=self.fixture();self.assertRaises(ValueError,recollect_loaded,rows[:-1],events,guards);rows[0]['started_epoch']=4;self.assertRaises(ValueError,recollect_loaded,rows,events,guards)
if __name__=='__main__':unittest.main()
