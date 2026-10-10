"""Exact typed canonical roundtrip control, authentic saved binding metadata."""
import copy,json,unittest
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from recollect_paired_binding_types48_v1 import differences
class Controls(unittest.TestCase):
 def test_tuple_and_intkeys_only_representation_roundtrip(self):
  actual={'schema':4,'metadata':{0:{'shape':(2,1),'ok':False}},'metrics':{'zero':-0.0,'nmse':0.0}};saved=json.loads(json.dumps(actual));self.assertNotEqual(actual,saved);self.assertEqual(canonical(actual),canonical(saved));types=differences(actual,saved);self.assertTrue(any(r.get('actual_key_type')=='int' for r in types));self.assertTrue(any(r.get('actual_type')=='tuple' for r in types))
 def test_bool_int_float_signedzero_changes_are_real(self):
  for a,b in [(True,1),(1,1.0),(-0.0,0.0),(False,0)]:self.assertNotEqual(canonical({'x':a}),canonical({'x':b}))
 def test_collision_nonfinite_duplicate_are_not_roundtrip_equivalence(self):
  for actual in ({1:'a','1':'b'},{'x':float('nan')},{'x':float('inf')}):self.assertRaises(ValueError,canonical,actual)
 def test_authentic_saved_plan_and_report_core_content(self):
  root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010');plan=json.loads((root/'batch46-pair-native4-off-prepared-v1/plan.json').read_text());report=json.loads((root/'private-native-offon-source37-v4-readonly-report-v1.json').read_text());binding=plan['paired_two_control_binding'];self.assertEqual(canonical(binding['actual_private_off_on_binding']),canonical(report));self.assertEqual(canonical(binding['actual_all49_serial']),canonical(report['actual_complete_serial49']['comparisons']))
 def test_authentic_full_binding_roundtrip_retains_all_fields(self):
  root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010');binding=json.loads((root/'batch46-pair-native4-off-prepared-v1/plan.json').read_text())['paired_two_control_binding'];actual=copy.deepcopy(binding);actual['CPU_tiny_repr_control']={0:(1,2)};saved=json.loads(json.dumps(actual));self.assertEqual(canonical(actual),canonical(saved));saved['engine_receipt_sha256']='0'*64;self.assertNotEqual(canonical(actual),canonical(saved))
if __name__=='__main__':unittest.main()
