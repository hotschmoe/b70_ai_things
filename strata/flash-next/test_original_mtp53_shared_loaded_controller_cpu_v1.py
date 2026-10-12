import unittest
from pathlib import Path
import build_original_mtp53_shared_loaded_v1 as b
import run_original_mtp53_shared_loaded_v1 as r
class Controllers(unittest.TestCase):
 def test_real_loaded_compile(self):
  script=b.shell({'compile_commands':[{'name':'original-mtp53-shared-slots','argv':['icpx','fixture.cpp','-o','/out/original-mtp53-shared-slots']} ]})
  self.assertIn('test ! -e /dev/dri',script);self.assertIn('icpx fixture.cpp',script)
 def test_logical_gpu1_and_full_identity(self):
  s=Path(r.__file__).read_text();self.assertIn('[(8,0),(9,1)]',s);self.assertIn('ZE_AFFINITY_MASK=0,1',s)
  self.assertIn("identity('pre')",s);self.assertIn("identity('post')",s)
  self.assertIn('before==after and digest==MTP_SHA',s);self.assertIn('xpu-collective-health',s)
  self.assertIn('str(MTP)+\':/original.gguf:ro\'',s)
  self.assertIn('packs=1 sessions=4 cells_each=5',r.MARKER)
