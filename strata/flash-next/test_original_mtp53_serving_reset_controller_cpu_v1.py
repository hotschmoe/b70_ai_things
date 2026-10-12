import unittest
import build_original_mtp53_serving_reset_v1 as b
import run_original_mtp53_serving_reset_v1 as r
class Controls(unittest.TestCase):
 def test_two_compile_outputs(self):
  recipe={'compile_commands':[{'name':'original-mtp53-serving-semantic.o','argv':['icpx','-c','bridge.cpp']},{'name':'original-mtp53-shared-reset','argv':['icpx','reset.cpp']}]}
  shell=b.shell(recipe);self.assertIn('icpx -c bridge.cpp',shell);self.assertIn('icpx reset.cpp',shell);self.assertIn('test ! -e /dev/dri',shell)
 def test_exact_reset_result(self):
  self.assertIn('reset_slots=2 fresh_references=2',r.MARKER);self.assertIn('private_reset_allocations_unchanged=1',r.MARKER)
