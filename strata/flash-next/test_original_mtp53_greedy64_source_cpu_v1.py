import unittest
from pathlib import Path
import prepare_original_mtp53_greedy64_compile_v1 as m
class GreedySource(unittest.TestCase):
 def test_actual_generated_dispatch_and_budget(self):
  s=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/sycl/src/program/generate.cpp').read_text();out=m.transform(s)
  self.assertIn('if(original53_mode)',out);self.assertIn('o.max_new!=64',out);self.assertIn('o.tokens.size()+64>256',out)
  self.assertIn('original_greedy53(original53.get(),ver,o.tokens,int(o.max_new)',out)
  self.assertIn('std::strcmp(original53_mode,"1")==0',out);self.assertIn('original53?2775258112ull:0ull',out)
 def test_actual_prompt_conditioning_no_peek_and_eos_commit(self):
  s=(m.HERE/'original-mtp53/sustained_greedy_v1.hpp').read_text();d=(m.HERE/'original-mtp53/draft_step_v3.hpp').read_text()
  self.assertIn('row+1<prompt.size()?int32_t(prompt[row+1]):first',s)
  self.assertNotIn('expected=',s);self.assertNotIn('conversation_checkpoint',s)
  self.assertIn('target_rows=remaining>=2?2:1',d);self.assertIn('keep=accepted && !first_eos?2:1',d)
  self.assertIn('generated<budget',s);self.assertIn('if(eos(actual[row])){stopped=true;break;}',s)
  for remaining,accepted,first_eos,want in [(1,False,False,1),(2,True,False,2),(2,True,True,1),(2,False,False,1)]:
   rows=2 if remaining>=2 else 1;keep=2 if rows==2 and accepted and not first_eos else 1;self.assertEqual(keep,want);self.assertLessEqual(keep,remaining)
