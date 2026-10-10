"""Tiny tmp ELF-like bytes/receipt fixtures; no SDK binary/model/pack reads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import operation_sdk37_byte_witness_v1 as w
class Controls(unittest.TestCase):
 def fixture(self,r):
  rows=[];receipts=[]
  for index,role in enumerate(w.ROLES):
   p=r/str(index);p.write_bytes(b'\x7fELF CPUfixture '+str(index).encode());rows.append((role,p,hashlib.sha256(p.read_bytes()).hexdigest()))
  for index in range(2):
   p=r/('receipt'+str(index));p.write_text(json.dumps({'CPU_metadata':index}));receipts.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))
  return rows,receipts
 def test_oversized_receipt_rejected_before_any_byte_read(self):
  with patch.object(w,'stat5',return_value=[1,2,17<<20,4,5]),patch.object(Path,'open',side_effect=AssertionError('must not read oversized receipt')) as opened:
   self.assertRaises(ValueError,w.metadata_bytes,Path('/CPU-synthetic-oversized'));opened.assert_not_called()
 def test_many_nested_calls_three_fresh_complete_reads_and_magic(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d))
   with patch.object(w,'read_elf',wraps=w.read_elf) as read:
    epoch=w.SDK37Epoch(rows,receipts)
    for n in range(80):epoch.require_current_roster(rows,receipts);self.assertEqual(epoch.digest_for(*rows[0]),rows[0][2]);self.assertEqual(epoch.ELF_magic(*rows[0]),b'\x7fELF')
    epoch.seal_predevice();epoch.require_current_roster(rows,receipts);epoch.digest_for(*rows[1]);proof=epoch.finalize();self.assertEqual(read.call_count,27);self.assertEqual(proof['unique_executables'],9);self.assertFalse(proof['stat_only_validation']);self.assertFalse(proof['source37_ELF_marker_scan_invented'])
 def test_payload_mutation_before_device_or_after_seal_refused(self):
  for after in (False,True):
   with tempfile.TemporaryDirectory() as d:
    rows,receipts=self.fixture(Path(d));epoch=w.SDK37Epoch(rows,receipts)
    if after:epoch.seal_predevice()
    rows[0][1].write_bytes(b'\x7fELF CHANGED');self.assertRaises(ValueError,epoch.finalize if after else epoch.seal_predevice)
 def test_receipt_mutation_and_consumer_digest_drift_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));epoch=w.SDK37Epoch(rows,receipts);receipts[0][0].write_text('{"changed":1}');self.assertRaises(ValueError,epoch.require_current_roster,rows,receipts);self.assertRaises(ValueError,epoch.seal_predevice)
 def test_missing_duplicate_or_wrong_roles_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d))
   for bad in (rows[:-1],rows+rows[:1],[('foreign',rows[0][1],rows[0][2])]+rows[1:]):self.assertRaises(ValueError,w.SDK37Epoch,bad,receipts)
 def test_nonELF_byte_matched_digest_still_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));rows[0][1].write_bytes(b'notELF');rows[0]=(rows[0][0],rows[0][1],hashlib.sha256(rows[0][1].read_bytes()).hexdigest());self.assertRaises(ValueError,w.SDK37Epoch,rows,receipts)
 def test_fork_owner_or_expiry_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));epoch=w.SDK37Epoch(rows,receipts)
   with patch.object(w.os,'getpid',return_value=epoch.owner+1):self.assertRaises(ValueError,epoch.seal_predevice)
   epoch.started-=901;self.assertRaises(ValueError,epoch.seal_predevice)
 def test_phase_closed_or_unsealed_no_current_proof(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));epoch=w.SDK37Epoch(rows,receipts);self.assertRaises(ValueError,epoch.finalize);epoch.seal_predevice();self.assertRaises(ValueError,epoch.seal_predevice);epoch.finalize();self.assertRaises(ValueError,epoch.digest_for,*rows[0]);self.assertRaises(ValueError,epoch.require_current_roster,rows,receipts)
 def test_saved_or_wrong_expected_sha_never_entry_admission(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));rows[0]=(rows[0][0],rows[0][1],'0'*64);self.assertRaises(ValueError,w.SDK37Epoch,rows,receipts)
 def test_unknown_semantic_role_path_or_expected_digest_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.fixture(Path(d));epoch=w.SDK37Epoch(rows,receipts)
   for bad in [('unknown',rows[0][1],rows[0][2]),(rows[0][0],rows[1][1],rows[0][2]),(rows[0][0],rows[0][1],'0'*64)]:self.assertRaises(ValueError,epoch.digest_for,*bad)
if __name__=='__main__':unittest.main()
