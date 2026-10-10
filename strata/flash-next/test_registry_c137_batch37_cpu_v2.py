"""Exact declared appendorder/FULL roster, no canonical modification/model probe."""
import tempfile,unittest
from pathlib import Path
import registry_c137_batch37_entry_association_v2 as q
class Controls(unittest.TestCase):
 def test_exact_full_appendorder_and_source_identity_preservation(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'models.yaml';p.write_bytes(q.BASELINE.read_bytes()+q.DELTA.read_bytes()+q.BATCH_DELTA.read_bytes());r=q.association(p);self.assertTrue(r['every_original_entry_byte_and_order_preserved']);self.assertEqual(len(r['C137_new_aliases']),4);self.assertEqual(len(r['batch37_new_aliases']),12);self.assertFalse(r['original_current_global_gates_passed']);self.assertFalse(r['C137_SDK_or_math_proof_transferred'])
 def test_subset_wrongorder_or_priorbyte_change_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'models.yaml'
   for raw in (q.BASELINE.read_bytes()+q.BATCH_DELTA.read_bytes()+q.DELTA.read_bytes(),q.BASELINE.read_bytes()+q.DELTA.read_bytes()+q.BATCH_DELTA.read_bytes()[:1000],q.BASELINE.read_bytes().replace(b'endpoint:',b'changed:',1)+q.DELTA.read_bytes()+q.BATCH_DELTA.read_bytes()):
    p.write_bytes(raw)
    with self.assertRaises(ValueError):q.association(p)
 def test_entry_specific_old_digest_is_not_claimed_current(self):
  import re
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);here=root/'strata/flash-next';here.mkdir(parents=True);canonical=root/'evals/configs/models.yaml';canonical.parent.mkdir(parents=True);canonical.write_bytes(q.BASELINE.read_bytes()+q.DELTA.read_bytes()+q.BATCH_DELTA.read_bytes());alias=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',q.BASELINE.read_text(),re.M)[0]
   with patch.object(q,'HERE',here):r=q.current_entry({'path':str(canonical),'sha256':q.BASELINE_SHA,'served_model_id':alias})
   self.assertTrue(r['entry_bytes_and_identity_preserved']);self.assertFalse(r['original_global_digest_still_current']);self.assertFalse(r['original_whole_global_gate_passed']);self.assertFalse(r['runtime_or_math_qualification_transferred'])
if __name__=='__main__':unittest.main()
