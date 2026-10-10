"""Small tmp byte fixtures only; never actual pack/model reads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import operation_pack_hash_witness_v2 as w
class Controls(unittest.TestCase):
 def roster(self,root):
  p=root/'tiny-pack';p.write_bytes(b'known CPU bytes');return [(p,hashlib.sha256(p.read_bytes()).hexdigest())]
 def test_many_semantic_calls_exactly_three_actual_read_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d))
   with patch.object(w,'current_row',wraps=w.current_row) as read:
    epoch=w.PackEpoch(roster);epoch.require_roster(roster)
    for i in range(80):self.assertEqual(epoch.digest_for(*roster[0]),roster[0][1])
    epoch.seal_predevice();proof=epoch.finalize();self.assertEqual(read.call_count,3);self.assertEqual(proof['semantic_digest_calls'],80);self.assertFalse(proof['stat_only_validation']);self.assertTrue(proof['between_boundaries_mutation_unobserved'])
 def test_mutation_between_start_and_predevice_refused(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));epoch=w.PackEpoch(roster);roster[0][0].write_bytes(b'changed');self.assertRaises(ValueError,epoch.seal_predevice)
 def test_mutation_after_predevice_refused_postoperation(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));epoch=w.PackEpoch(roster);epoch.seal_predevice();roster[0][0].write_bytes(b'changed');self.assertRaises(ValueError,epoch.finalize)
 def test_current_roster_drift_subset_duplicate_refused(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));epoch=w.PackEpoch(roster)
   for bad in ([],roster+roster,[(roster[0][0],'0'*64)]):self.assertRaises(ValueError,epoch.require_roster,bad)
 def test_wrong_file_expected_digest_phase_refused(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));epoch=w.PackEpoch(roster);self.assertRaises(ValueError,epoch.digest_for,roster[0][0],'0'*64);epoch.seal_predevice();self.assertEqual(epoch.digest_for(*roster[0]),roster[0][1]);self.assertRaises(ValueError,epoch.seal_predevice);epoch.finalize();self.assertRaises(ValueError,epoch.digest_for,*roster[0])
 def test_saved_digest_is_never_initial_current_proof(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));roster[0][0].write_bytes(b'changed');self.assertRaises(ValueError,w.PackEpoch,roster)
 def test_process_ownership_and_expiry(self):
  with tempfile.TemporaryDirectory() as d:
   epoch=w.PackEpoch(self.roster(Path(d)))
   with patch.object(w.os,'getpid',return_value=epoch.owner+1):self.assertRaises(ValueError,epoch.seal_predevice)
   epoch.started-=901;self.assertRaises(ValueError,epoch.seal_predevice)
 def test_explicit_current_prepared_pack_adapter(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);roster=self.roster(root);receipt=root/'receipt';receipt.write_text(json.dumps({'RESULT':{'files':{'tiny-pack':{'sha256':roster[0][1]}}}}));prepared={'pack':str(root),'pack_receipt':str(receipt),'pack_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()};epoch=w.PackEpoch(roster);self.assertEqual(w.consume_prepared_pack(prepared,epoch)['members'],1);epoch.seal_predevice();epoch.finalize()
 def test_adapter_changed_current_receipt_or_roster_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);roster=self.roster(root);receipt=root/'receipt';receipt.write_text(json.dumps({'RESULT':{'files':{'tiny-pack':{'sha256':'0'*64}}}}));prepared={'pack':str(root),'pack_receipt':str(receipt),'pack_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()};epoch=w.PackEpoch(roster);self.assertRaises(ValueError,w.consume_prepared_pack,prepared,epoch);receipt.write_text('{}');self.assertRaises(ValueError,w.consume_prepared_pack,prepared,epoch)
 def test_close_requires_predevice_and_cannot_reuse(self):
  with tempfile.TemporaryDirectory() as d:
   roster=self.roster(Path(d));epoch=w.PackEpoch(roster);self.assertRaises(ValueError,epoch.finalize);epoch.seal_predevice();epoch.finalize();self.assertRaises(ValueError,epoch.finalize)
if __name__=='__main__':unittest.main()
