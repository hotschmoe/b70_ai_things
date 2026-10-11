import ast,unittest
from unittest.mock import patch
import indexer_materialized41_masked_inputs_v2 as inputs
import admit_postboot_original_qsa_reference_v4 as reader

class SuccessorTests(unittest.TestCase):
 def test_actual_nested_source_preflight(self):
  self.assertEqual(len(reader.source_binding()),64)
 def test_preflight_refusal_before_byte_epoch(self):
  inputs._operation=None
  with patch.object(reader,'source_binding',side_effect=ValueError('source refused')),patch('native_qsa40_byte_epoch_v9.ByteEpoch')as epoch:
   with self.assertRaisesRegex(ValueError,'source refused'):inputs.producer_binding(inputs.OWN)
   epoch.assert_not_called()
 def test_fresh_independent_admission_path(self):
  self.assertEqual(inputs.ADMISSION.name,'postboot-original-qsa-reference-admission-v4-current173.json')
  self.assertNotEqual(inputs.ADMISSION.name,'postboot-original-qsa-reference-admission-v3.json')
 def test_explicit_registry_before_current_baseline(self):
  tree=ast.parse(reader.HERE.joinpath('admit_postboot_original_qsa_reference_v4.py').read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='finalized_binding')
  text=ast.unparse(fn)
  self.assertLess(text.index('registry_association('),text.index('current_baseline(current_c140_root)'))
  self.assertIn('registry_entry_preservation',text)

if __name__=='__main__':unittest.main()
