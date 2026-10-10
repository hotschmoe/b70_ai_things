"""Producer-shaped AST and observed epoch negatives; no model/helper execution."""
import unittest,copy
from pathlib import Path
import first_hc_computation_chronology_v2 as c
HERE=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_actual_v2_source(self):self.assertTrue(c.source_contract(HERE/'explore_owned_first_hc_fidelity_v2.py')['source_order_validated'])
 def test_actual_old_early_terminal_rejected(self):
  with self.assertRaises(ValueError):c.source_contract(HERE/'explore_owned_first_hc_fidelity_v1.py')
 def report(self):return dict(conditional_captured_inputs_requested=True,owned_HC_terminal_epoch=10.,conditional_GDN_terminal_epoch=20.,computation_terminal_epoch=21.)
 def test_actual_shape_good(self):self.assertEqual(c.observed(self.report())['computation_terminal_epoch'],21.)
 def test_early_terminal(self):
  r=self.report();r['computation_terminal_epoch']=15.
  with self.assertRaises(ValueError):c.observed(r)
 def test_unobserved_nonfinite(self):
  for val in (True,float('nan'),float('inf'),-1.):
   r=self.report();r['conditional_GDN_terminal_epoch']=val
   with self.assertRaises(ValueError):c.observed(r)
 def test_option_off(self):
  r=self.report();r['conditional_captured_inputs_requested']=False;r.pop('conditional_GDN_terminal_epoch');c.observed(r)
if __name__=='__main__':unittest.main()
