import unittest
from source43_cache_port_recipe_v1 import proposal,recipe
class Controls(unittest.TestCase):
 def cfg(self):return {'args':['--max-context','2048','--prefill','64','--batch','0'],'env':{'STRATA_QSA_INDEXER_MATERIALIZED_HALF':'1'},'parallel':1}
 def test_actual_population_not_selected_RIDs(self):
  for n in(1,2,4,6):
   value=proposal(self.cfg(),n,[1]);self.assertEqual(value['target_actual_active_population'],n);self.assertEqual(value['native_batch_capacity'],max(2,n));self.assertFalse(value['API_launch_allowed']);self.assertEqual(value['configuration_proposal']['parallel'],max(2,n));self.assertEqual(value['configuration_proposal']['env']['STRATA_BATCH_CHAIN_MIB'],'8192')
 def test_no_strict_batch1_or_typed_float(self):
  for n in(1,True,2.0,8):self.assertRaises(ValueError,recipe,self.cfg(),n,[1])
 def test_mode0_and_duplicate_RID_refused(self):
  cfg=self.cfg();cfg['env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF']='0';self.assertRaises(ValueError,recipe,cfg,2,[1]);self.assertRaises(ValueError,recipe,self.cfg(),2,[1,1])
if __name__=='__main__':unittest.main()
