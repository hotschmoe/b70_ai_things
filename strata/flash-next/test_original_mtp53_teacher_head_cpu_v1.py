import unittest
from pathlib import Path
H=Path(__file__).resolve().parent/'original-mtp53'
class Teacher(unittest.TestCase):
 def test_recurrence_and_decode_unchanged(self):
  old=(H/'draft_step_pair_v1.hpp').read_text();new=(H/'draft_step_teacher_head_v1.hpp').read_text()
  start='        eh_hc_front(device_token';end='        hc_read53(s.next_hidden'
  self.assertIn(old[old.index(start):old.index(end)],new)
  oldtail=old[old.index('    // Synthetic fixture calls'):]
  self.assertTrue(new.endswith(oldtail))
  self.assertIn('bool compute_head=true)',new)
 def test_only_teacher_dispatch_skips(self):
  old=(H/'loaded_main_adapter_pair_v3.hpp').read_text();new=(H/'loaded_main_adapter_teacher_head_v1.hpp').read_text()
  self.assertEqual(new.count('!teacher_head_skip_'),1)
  self.assertIn('catchup(int32_t(next),rows+size_t(t)*10240,!teacher_head_skip_);++teacher_rows_',new)
  self.assertIn('try{catchup(next,target_hidden);}',new)
  self.assertEqual(old[old.index(' bool exercise_pair('):],new[new.index(' bool exercise_pair('):])
  self.assertIn('if(!draft->head_ready())throw',new)
  greedy=(H/'paired_greedy_v2.hpp').read_text()
  self.assertLess(greedy.index('resident.advance_actual(first'),greedy.index('resident.propose_actual()'))
 def test_head_has_no_recurrence_destination(self):
  old=(H/'draft_step_pair_v1.hpp').read_text();line=next(l for l in old.splitlines()if 'hc_read53(s.next_hidden,w.head_hc' in l)
  self.assertNotIn('state_',line);self.assertNotIn('committed_hidden_',line)
  self.assertIn('s.head_q8,s.logits',line)
  # The next front uses actual MAIN hidden, never head logits or argmax.
  adapter=(H/'loaded_main_adapter_teacher_head_v1.hpp').read_text()
  self.assertIn('draft->record(token,hidden,W,S,compute_head)',adapter)
