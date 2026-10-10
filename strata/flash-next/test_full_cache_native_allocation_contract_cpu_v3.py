import copy,unittest
from full_cache_native_allocation_contract_v3 import analyze

class Accounting(unittest.TestCase):
 def fixture(self):
  owner={'engine_pid':17,'process_start_ticks':20,'device_uuid':'physical-card-0','source_generation':39};context={'current_main_body_rid':3,'actual_slot_rid':4,'phase':'BSTEP'}
  rows=[{'kind':'allocation_success','allocation_id':'arena1','space':'device_usm','bytes':64,'api_result':'success'},{'kind':'expert_bytes_ready','allocation_id':'arena1','layer':1,'expert':2,'offset':0,'bytes':32,'copy_completion_observed':True,'existing_completion_boundary':True},{'kind':'expert_bytes_released','allocation_id':'arena1','layer':1,'expert':2},{'kind':'allocation_release_success','allocation_id':'arena1','api_result':'success'}]
  for i,r in enumerate(rows,1):r.update(owner=copy.deepcopy(owner),sequence=i,host_epoch=float(i),command_context=copy.deepcopy(context))
  return owner,rows
 def test_api_allocation_peak_not_expert_double_count_or_residency(self):
  owner,rows=self.fixture();r=analyze(rows,owner,[[1,2]]);self.assertEqual(r['peak_live_API_bytes_by_space']['device_usm'],64);self.assertFalse(r['per_expert_physical_residency_qualified']);self.assertFalse(r['current_native_producer_or_runtime_qualified'])
 def test_owner_sequence_copy_and_extent_fail_closed(self):
  for index,key,value in [(0,'sequence',2),(1,'offset',40),(1,'copy_completion_observed',False),(1,'existing_completion_boundary',False),(3,'api_result','failure')]:
   with self.subTest(key=key):
    owner,rows=self.fixture();rows[index][key]=value
    with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_no_free_before_expert_retirement(self):
  owner,rows=self.fixture();rows[2],rows[3]=rows[3],rows[2]
  for i,r in enumerate(rows,1):r.update(sequence=i,host_epoch=float(i))
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_no_unretired_end_or_physical_invention(self):
  owner,rows=self.fixture()
  with self.assertRaises(ValueError):analyze(rows[:-1],owner,[[1,2]])
  rows[1]['kind']='physical_resident'
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_changed_actual_slot_context_cannot_be_omitted(self):
  owner,rows=self.fixture();del rows[1]['command_context']['actual_slot_rid']
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_no_expert_events_cannot_claim_coverage(self):
  owner,rows=self.fixture();rows=[rows[0],rows[-1]];rows[-1]['sequence']=2
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_declared_missing_expert_and_out_of_range_rejected(self):
  owner,rows=self.fixture()
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2],[1,3]])
  rows[1]['expert']=512
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_float_generation_and_empty_roster_rejected(self):
  owner,rows=self.fixture();owner['source_generation']=39.0
  with self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
  owner,rows=self.fixture()
  with self.assertRaises(ValueError):analyze(rows,owner,[])
 def test_actual_row_owner_float_and_release_key_float_rejected(self):
  for mutate in ('owner','release'):
   owner,rows=self.fixture()
   if mutate=='owner':rows[-1]['owner']['source_generation']=39.0
   else:rows[2]['expert']=2.0
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
 def test_unbacked_virtual_or_physical_handle_cannot_be_expert_bytes(self):
  for space in ('virtual_address','physical_handle','host_mapping'):
   owner,rows=self.fixture();rows[0]['space']=space
   with self.subTest(space=space),self.assertRaises(ValueError):analyze(rows,owner,[[1,2]])
if __name__=='__main__':unittest.main()
