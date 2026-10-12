import unittest
from pathlib import Path
H=Path(__file__).resolve().parent/'original-mtp53'
class Bridge(unittest.TestCase):
 def test_reset_has_no_device_allocation(self):
  s=(H/'shared_weights_sessions_v4.hpp').read_text();body=s[s.index(' void reset_fresh('):s.index(' // Bounded diagnostic ONLY')]
  self.assertNotIn('owner.take',body);self.assertNotIn('malloc_',body)
  for field in ['committed_bank_','working_bank_','committed_h','has_logits_=false','slot_=next']:self.assertIn(field,body)
  self.assertLess(body.index('q.wait_and_throw();'),body.index('zero(committed_bank_)'))
 def test_actual_commit_before_all_terminal_catchup(self):
  s=(H/'serving_slot_bridge_v1.hpp').read_text();body=s[s.index(' void commit_and_catchup'):s.index(' bool request_cancel')]
  self.assertLess(body.index('first_->commit_slot_prefixes'),body.index('s.session->advance_actual'))
  self.assertNotIn('if(s.live)',body);self.assertIn('r.first_row+j',body)
  self.assertNotIn('require_last',body);self.assertNotIn('main.run',body)
  self.assertIn('refuse_legacy_cache_restore',s)
