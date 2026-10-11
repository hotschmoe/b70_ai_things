import unittest
import source43_num10_zero_case_v1 as c
import prefill_chunk64_512_control_v2 as template
class Controls(unittest.TestCase):
 def fixture(self):
  cfg=template.expected_config('chunk64',{'inherited_math_env':{}},18082);return {'cards':[0],'alias':cfg['aliases'][0]},cfg
 def test_exact_four_literal_prefix1_and_observer_exclusion(self):
  m,cfg=self.fixture();on=c.recipe(m,cfg,'a'*64,True);off=c.recipe(m,cfg,'a'*64,False);self.assertEqual(on['requests'],off['requests']);self.assertEqual(len(on['requests']),4);self.assertTrue(all(r['ids']==[248045]and r['pin']is None and r['max_new']==1 for r in on['requests']));self.assertEqual(on['native_env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'1');self.assertEqual(on['native_env'].get('STRATA_QSA3_TARGET','0'),'0');self.assertEqual(on['native_env']['STRATA_LAYER0_Q8_DIAG'],'1');self.assertEqual(off['native_env']['STRATA_LAYER0_Q8_DIAG'],'0');self.assertEqual(on['native_env']['SYCL_UR_TRACE'],'2')
 def test_eager_qsa_or_bool_arm_refused(self):
  m,cfg=self.fixture()
  with self.assertRaises(ValueError):c.recipe(m,cfg,'a'*64,1)
  cfg['env']['STRATA_QSA3_TARGET']='1'
  with self.assertRaises(ValueError):c.recipe(m,cfg,'a'*64,True)
if __name__=='__main__':unittest.main()
