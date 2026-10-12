import unittest,tempfile,json
from pathlib import Path
import run_original_mtp53_expert_pair_model_v3 as r
class HistoricalOFF(unittest.TestCase):
 def test_foreign_report_refuses_before_native_intake(self):
  original=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-mtp53-expert-pair-model-card0-run-v1/report.json')
  data=r.read(original)
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'report.json').write_bytes(original.read_bytes())
   with self.assertRaises(ValueError):r.historical_off(root,'0'*64,data['compile_receipt_sha256'],data['elf_sha256'])
   self.assertEqual([x.name for x in root.iterdir()],['report.json'])
 def test_historical_scope_and_closing_old_evidence(self):
  s=Path(r.__file__).read_text();self.assertIn("'original_parent_passed':False",s);self.assertIn("'new_runtime_admission_claimed':False",s)
  self.assertIn("saved_off['consumed'].items()",s);self.assertIn("else:arm(0)\n  arm(1)",s)
  self.assertIn('Complete original profile permutations required',s)
