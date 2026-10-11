import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_num10_original_reference_v3 as q
import test_source43_num10_original_reference_cpu_v1 as first
import test_source43_num10_original_reference_cpu_v2 as previous
class Controls(first.Controls):
 def setUp(self):self.p=patch.object(first,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class RawControls(first.RawControls):
 def setUp(self):self.p=patch.object(first,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
 def fixture(self,root,kind):
  b,rows=super().fixture(root,kind);b['binding_sha256']=b['plan_sha256'];return b,rows
class ClosingControls(first.ClosingControls):
 def setUp(self):self.p=patch.object(first,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class NewControls(previous.NewControls):
 def setUp(self):self.p=patch.object(previous,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
 def test_arithmetic_and_target_functions_byte_semantics_unchanged(self):
  source=Path(q.__file__).with_name('source43_num10_original_reference_v2.py').read_text()
  expected=source.replace('source43-num10-original-reference-source-plan-v2.json','source43-num10-original-reference-source-plan-v3.json').replace("frame['binding_sha256']==binding['plan_sha256']","frame['binding_sha256']==binding['binding_sha256']")
  self.assertEqual(Path(q.__file__).read_text(),expected)

class BindingControls(RawControls):
 # Only the two new cases; inherited five controls run once above.
 test_actual_shaped_raw33=None;test_nonzero_native_conv=None;test_native_negative_zero_not_initial_bytezero=None;test_wrong_actual_accepted_token=None;test_nonfresh_actual_request=None
 def test_observer_hash_distinct_from_corpus_plan_hash(self):
  with tempfile.TemporaryDirectory()as t:
   b,rows=self.fixture(Path(t),'valid');b['plan_sha256']='c'*64;self.assertNotEqual(b['plan_sha256'],b['binding_sha256']);self.assertEqual(len(q.load_targets(b,rows)),4)
 def test_frame_falsely_rebound_to_corpus_hash_refused(self):
  with tempfile.TemporaryDirectory()as t:
   b,rows=self.fixture(Path(t),'valid');b['plan_sha256']='c'*64
   for row in rows:
    f=row['layer0']['frame'];f['binding_sha256']=b['plan_sha256'];p=Path(row['layer0']['metadata']);p.write_text(json.dumps(f));row['layer0']['metadata_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaisesRegex(ValueError,'owner/request/nonce'):q.load_targets(b,rows)
if __name__=='__main__':unittest.main()
