import unittest
from pathlib import Path
import prepare_original_mtp53_shared_compile_v1 as p
class Compile(unittest.TestCase):
 def test_exact_context_delta(self):
  h=p.HERE/'original-mtp53';old=(h/'shared_weights_sessions_v1.hpp').read_text();new=(h/'shared_weights_sessions_v2.hpp').read_text()
  delta=' || q.get_context()!=dpct::get_in_order_queue().get_context()'
  self.assertEqual(new.replace(delta,'',1),old)
  self.assertEqual(new.count(delta),1)
 def test_compile_only(self):
  r={'compile_commands':[{'argv':['icpx','-fp-model=precise','/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp','-Wl,--start-group','old.a','-o','old']} ]}
  cmd=p.compile_command(r);self.assertIn('-c',cmd);self.assertIn('-fp-model=precise',cmd);self.assertNotIn('old.a',cmd)
  self.assertEqual(cmd[-1],'/out/original-mtp53-shared-sessions.o')
  self.assertIn('original_mtp53_shared_sessions_compile.cpp',cmd[-3])
