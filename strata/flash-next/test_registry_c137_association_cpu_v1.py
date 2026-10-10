"""Exact named append association, no actual registry edit or model/runtime probe."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import registry_c137_entry_association_v1 as q
class Controls(unittest.TestCase):
 def test_exactdeclared_C137append_preserves_oldgate_false(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'models.yaml';path.write_bytes(q.BASELINE.read_bytes()+q.DELTA.read_bytes());r=q.association(path);self.assertTrue(r['every_original_entry_byte_and_order_preserved']);self.assertFalse(r['original_current_global_gates_passed']);self.assertFalse(r['C137_SDK_or_math_proof_transferred']);self.assertEqual(len(r['C137_new_aliases']),4)
 def test_changed_oldbyte_or_extraalias_or_reorderedappend_refused(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'models.yaml'
   for raw in (q.BASELINE.read_bytes().replace(b'endpoint:',b'changed:',1)+q.DELTA.read_bytes(),q.BASELINE.read_bytes()+q.DELTA.read_bytes()+b'EXTRA\n',q.DELTA.read_bytes()+q.BASELINE.read_bytes()):
    path.write_bytes(raw)
    with self.assertRaises(ValueError):q.association(path)
 def test_namedV7view_changes_only_actualregistrydigest(self):
  import batch_numerical_execution_v7 as frozen
  plan={'registry_binding':{'sha256':q.BASELINE_SHA,'CPU_SYNTHETIC':True},'actual_raw_original':'unchanged'};before=copy.deepcopy(plan);seen=[];proof={'actual_current_global_sha256':'CPU_NEW_ACTUAL_DIGEST'}
  def gate(view):seen.append(copy.deepcopy(view));return {'CPU_OTHER_GATES':True}
  with patch.object(q,'historical_entry',return_value=proof),patch.object(frozen,'manifest_binding',gate):r=q.admit_historical_v7_plan(plan)
  self.assertEqual(plan,before);expected=copy.deepcopy(before);expected['registry_binding']['sha256']='CPU_NEW_ACTUAL_DIGEST';self.assertEqual(seen,[expected]);self.assertFalse(r['original_current_global_gate_passed']);self.assertFalse(r['measurements_requalified'])
if __name__=='__main__':unittest.main()
