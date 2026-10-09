#!/usr/bin/env python3
"""Synthetic real-role zero-input ablation controls; no actual model payload/GPU."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from full48_owned_composition_storage_v1 import OwnedPle,OriginalStorageProjector,Full48OwnedComposition
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
from full48_zero_staged_ple_ablation_v1 import ZeroStagedPle,ZeroStagedPleComposition,PROVENANCE

class AblationTests(unittest.TestCase):
 def setUp(self):self.p=SyntheticOriginalRoles();self.project=OriginalStorageProjector(self.p)
 def test_original_table_read_but_projection_zero_explicit(self):
  ple=ZeroStagedPle(self.p,self.project,'a'*64);residual=np.linspace(-.2,.3,10240).reshape(4,2560);result=ple.advance(248045,residual);self.assertTrue(np.array_equal(result['embedding_F32'],np.zeros(2560)));self.assertGreater(np.count_nonzero(result['original_table_embedding_F32']),0);self.assertTrue(np.array_equal(result['postprojection']['result'],residual.astype('<f4').astype(np.float64)));self.assertEqual(np.count_nonzero(ple.history),0);self.assertFalse(result['reference_ablation']['actual_zero_native_PLE_measured'])
 def test_real_original_ple_not_relabelled_zero(self):
  residual=np.linspace(-.2,.3,10240).reshape(4,2560);normal=OwnedPle(self.p,self.project,'a'*64).advance(248045,residual);self.assertGreater(np.count_nonzero(normal['postprojection']['gated']),0);self.assertNotIn('reference_ablation',normal)
 def test_repeated_zero_stage_owns_prev_history(self):
  ple=ZeroStagedPle(self.p,self.project,'a'*64)
  for token in (19,22,23):ple.advance(token,np.ones((4,2560))*.02)
  self.assertEqual(ple.last_two,[22,23]);self.assertEqual(len(ple.tokens),3);self.assertEqual(np.count_nonzero(ple.history),0)
 def test_actual_role_full48_synthetic_ablation_keeps_layer0_changes_layer1(self):
  import gc
  normal=Full48OwnedComposition(self.p,'a'*64).tokens([19]);first=normal['trace'][0]['layers'][0]['ffn'].copy();second=normal['trace'][0]['layers'][1]['ffn'].copy();del normal;gc.collect();modified=ZeroStagedPleComposition(self.p,'a'*64).tokens([19]);self.assertTrue(np.array_equal(first,modified['trace'][0]['layers'][0]['ffn']));self.assertFalse(np.array_equal(second,modified['trace'][0]['layers'][1]['ffn']));self.assertFalse(modified['actual_exact_original_model_reference']);self.assertFalse(modified['full_model_math_qualified']);self.assertFalse(modified['reference_ablation']['actual_zero_native_PLE_measured'])
 def test_every_failed_report_labelled_and_original_driver_restored(self):
  from unittest.mock import patch
  import explore_full48_zero_staged_ple_prefix1_v1 as entry
  import explore_full48_original_prefix1_v1 as original
  old_composition=original.Full48OwnedComposition;old_write=original.write;old_binding=original.dependency_binding
  with tempfile.TemporaryDirectory() as temp:
   file=Path(temp)/'report.json'
   def failed_main():
    original.write(file,{'status':'CPU synthetic admission failure','numeric_pass_claim':False});return 1
   with patch.object(original,'main',side_effect=failed_main):self.assertEqual(entry.main(),1)
   result=original.read(file);self.assertFalse(result['reference_ablation']['actual_zero_native_PLE_measured']);self.assertFalse(result['reference_ablation']['actual_exact_original_model_reference']);self.assertEqual(result['modified_reference_driver_sha256'],original.sha(entry.__file__))
  self.assertIs(original.Full48OwnedComposition,old_composition);self.assertIs(original.write,old_write);self.assertIs(original.dependency_binding,old_binding)
 def test_provenance_never_claims_actual_math(self):
  self.assertFalse(PROVENANCE['actual_exact_original_model_reference']);self.assertFalse(PROVENANCE['native_math_or_model_qualification']);self.assertFalse(PROVENANCE['captured_values_used_as_reference_feed']);self.assertIn('MODIFIED_REFERENCE',PROVENANCE['kind'])

if __name__=='__main__':unittest.main()
