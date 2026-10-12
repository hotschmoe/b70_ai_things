import unittest
import build_original_mtp53_teacher_head_v1 as b
class Builder(unittest.TestCase):
 def test_single_native_recipe(self):
  r={'compile_commands':[{'name':'original-mtp53-two-stage','argv':['icpx','source','-o','/out/original-mtp53-two-stage']} ]}
  script=b.shell(r);self.assertIn('test ! -e /dev/dri',script);self.assertIn('icpx source -o',script)
 def test_absence_exact_owner(self):
  self.assertTrue(b.absent(b'error: no such object: owned\n','owned'))
  self.assertFalse(b.absent(b'error: no such object: other\n','owned'))
