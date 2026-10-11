import unittest
from pathlib import Path
import probe_pti_paused_regular_fixture_v2 as p
class Tests(unittest.TestCase):
 def test_only_cpp_header_name_changed(self):
  old=(p.HERE/'pti_paused_regular_list_fixture_v1.cpp').read_bytes();new=(p.HERE/'pti_paused_regular_list_fixture_v2.cpp').read_bytes();self.assertEqual(new,old.replace(b'sycl/ext/oneapi/experimental/command_graph.hpp',b'sycl/ext/oneapi/experimental/graph.hpp'))
 def test_real_compile_recipe_uses_new_source(self):
  cmd=p.compile_recipe(Path('/tmp/properties.json'))('compile',Path('/tmp/out'),'owned','a'*64);self.assertIn(str(p.HERE/'pti_paused_regular_list_fixture_v2.cpp')+':/leaf/fixture.cpp:ro',cmd);self.assertNotIn('--device',cmd);self.assertIn('-fsycl -O2 -std=c++17 -fsycl-targets=spir64',cmd[-1])
if __name__=='__main__':unittest.main()
