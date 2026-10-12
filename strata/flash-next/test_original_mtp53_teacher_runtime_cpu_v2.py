import unittest
from pathlib import Path
import run_original_mtp53_teacher_head_v2 as r
class Admission(unittest.TestCase):
 def test_real_source_equivalence(self):
  proof=r.unchanged_decode_source();self.assertTrue(proof['unchanged_decode_settle_source_verified']);self.assertFalse(proof['forced_teacher_controls_executed'])
 def test_baseline_before_candidate_and_no_forced_prerequisite(self):
  source=Path(r.__file__).read_text();self.assertIn("for mode in('off','on'):",source)
  self.assertIn("if len(parsed['ids'])!=512 or parsed['ids']!=original_reference['ids']",source)
  self.assertIn("if errors:raise ValueError('Arm'+str(mode)",source)
  self.assertNotIn('prior=read(a.controls_report)',source)
  self.assertIn("choices=['natural']",source)
