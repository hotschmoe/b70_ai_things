"""Synthetic tiny interface/trajectory controls, no helper or model runtime."""
import inspect,tempfile,json,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
import owned_first_hc_fidelity_v1 as f
import compare_original48_hc_trajectories_v1 as t
class Tests(unittest.TestCase):
 def test_original_only_interface(self):self.assertEqual(list(inspect.signature(f.compute_owned).parameters),['model','token_ids'])
 def test_token_roster(self):
  for ids in ([True],[0,1],[-1],[248320],[]):
   with self.assertRaises(ValueError):f.contract(ids)
 def test_conditional_explicit_zero(self):
  class G:
   def initial_state(self):return {'zero':True}
   def mixer(self,mixed,state,route):
    self.seen=(mixed,state,route);return {'state':np.zeros(1)},{'output':np.zeros(2560)}
  g=G();m=type('M',(),{'gdn':{0:g}})();row=f.conditional_gdn(m,np.zeros(2560));self.assertTrue(row['captured_inputs_used']);self.assertFalse(row['captured_states_used']);self.assertFalse(row['full_model_math_qualified']);self.assertEqual(g.seen[1],{'zero':True});self.assertEqual(g.seen[2],'verifier')
 def test_conditional_shape(self):
  with self.assertRaises(ValueError):f.conditional_gdn(None,np.zeros(32))
 def reports(self):
  comparisons={str(i):{'nmse':0.,'bitwise_equal':True,'numeric_gate_assigned':False} for i in range(577)};return {'errors':[],'native_observation_binding':{'actual':'CPU_SYNTHETIC'},'exploration':{'comparisons':comparisons}}
 def test_trajectory_exact_target(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'prior.json';c=Path(root)/'candidate.json';row=self.reports();p.write_text(json.dumps(row));c.write_text(json.dumps(row));self.assertEqual(len(t.compare(p,c)['rows']),577);row['native_observation_binding']={'different':'CPU'};c.write_text(json.dumps(row))
   with self.assertRaises(ValueError):t.compare(p,c)
 def test_no_threshold_or_failed_run(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'prior.json';c=Path(root)/'candidate.json';row=self.reports();p.write_text(json.dumps(row));row['exploration']['comparisons']['0']['numeric_gate_assigned']=True;c.write_text(json.dumps(row))
   with self.assertRaises(ValueError):t.compare(p,c)
 def test_source(self):f.source_binding()
if __name__=='__main__':unittest.main()
