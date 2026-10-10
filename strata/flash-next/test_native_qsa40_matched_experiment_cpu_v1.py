import copy,unittest
import native_qsa40_matched_experiment_v1 as p
class Controls(unittest.TestCase):
 def test_exact_topologies_and_quotas(self):
  for cards,zero in [([0],0),([0,1],12)]:
   r=p.recipe(cards);self.assertEqual(r['ON_nonowner_zero_frames'],zero);self.assertEqual(r['head_all48_lastrow_bitwise_pairs'],196);self.assertEqual(r['wholeprefix_P30_phase_bitwise_pairs'],2304);p.admit_roster(r['requests'])
  self.assertRaises(ValueError,p.recipe,[True]);self.assertRaises(ValueError,p.recipe,[1])
 def test_roster_refuses_old_prefixes_and_pin_or_fresh_changes(self):
  for key,value in [('ids',p.IDS[:1]),('pin',0),('fresh',False),('max_new',64),('ordinal',2)]:
   rows=p.request_roster();rows[0][key]=value;self.assertRaises(ValueError,p.admit_roster,rows)
  self.assertRaises(ValueError,p.admit_roster,p.request_roster()[:3])
 def test_exact_observer_delta_and_eager_absence(self):
  binding='a'*64;off=p.arm_environment({},False,binding,'/results/qsa');on=p.arm_environment({},True,binding,'/results/qsa');self.assertTrue(p.admit_arm_pair(off,on,binding,'/results/qsa'))
  altered=copy.deepcopy(on);altered['STRATA_HC_ROUTE']='other';self.assertRaises(ValueError,p.admit_arm_pair,off,altered,binding,'/results/qsa')
  for key in p.ABSENT:self.assertRaises(ValueError,p.arm_environment,{key:'0'},True,binding,'/results/qsa')
 def test_missing_internal_witness_never_substituted(self):
  r=p.recipe([0]);self.assertFalse(r['captures_are_math_inputs']);self.assertFalse(r['runtime_ready']);self.assertFalse(r['source37_runtime_proof_transferred']);self.assertNotIn('q_normalized.f32',r['device_target_mapping']);self.assertIn('internal_softmax',r['unobserved_native_values'])
if __name__=='__main__':unittest.main()
