import unittest
import build_original_mtp53_shared_semantic_v1 as b
class Builder(unittest.TestCase):
 def test_only_one_object(self):
  r={'compile_commands':[{'name':'original-mtp53-shared-sessions.o','argv':['icpx','-c','fixture.cpp','-o','/out/original-mtp53-shared-sessions.o']} ]}
  self.assertIn('test ! -e /dev/dri',b.shell(r));self.assertIn('icpx -c fixture.cpp',b.shell(r))
  r['compile_commands'][0]['argv'].remove('-c')
  with self.assertRaises(ValueError):b.shell(r)
