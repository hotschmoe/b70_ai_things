import unittest
from pathlib import Path
import prepare_original_mtp53_main_compile_v3 as m
class Diagnostic(unittest.TestCase):
 def test_output_only_delta_and_preassert_order(self):
  old=(m.HERE/'original-mtp53/loaded_main_adapter_v1.hpp').read_text();new=(m.HERE/'original-mtp53/loaded_main_adapter_v2.hpp').read_text()
  rows=new.splitlines(keepends=True);extra=[r for r in rows if 'MTP53_MAIN_PREASSERT' in r]
  self.assertEqual(len(extra),1);self.assertEqual(new.replace(extra[0],''),old)
  self.assertLess(new.index('MTP53_MAIN_PREASSERT'),new.index('if(actual[0]!=expected'))
  self.assertIn('control_acceptance_equal=%d',extra[0]);self.assertIn('base_position=%lld',extra[0])
 def test_versioned_header_and_singlecard_guard(self):
  text=Path(m.__file__).read_text();self.assertIn('loaded_main_adapter_v2.hpp',text)
  self.assertIn('!stages.empty()',text);self.assertNotIn('n_stages!=1',text)
