import unittest
from pathlib import Path
import original_mtp53_serve_baseline_history_v1 as h
import prepare_original_mtp53_serve_source_v3 as old
import prepare_original_mtp53_serve_source_v5 as new
class Default0(unittest.TestCase):
 def test_complete_actual_source_normalization_and_negative(self):
  source=(old.SDK/'source/sycl/src/program/generate.cpp').read_text();a=old.construct(source);b=new.construct(source);v=(Path(h.HERE)/'original-mtp53/paired_slot_verify_v1.cpp').read_text();base=(old.SDK/'source/sycl/src/core/verify.cpp').read_text();self.assertTrue(h.default0_source_equivalence(a,b,v,base)['default0_verifier_body_unchanged'])
  with self.assertRaises(ValueError):h.default0_source_equivalence(a,b.replace('            int64_t resume = 0;','            int64_t resume = 1;',1),v,base)
