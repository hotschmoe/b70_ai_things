import unittest
from pathlib import Path
import pti_unitrace_build_recipe_v1 as p
class Controls(unittest.TestCase):
 def test_whole_fresh_source_manifest(self):self.assertEqual(p.source_binding(),p.MANIFEST_SHA)
 def test_minimal_no_device_recipe_and_pinned_dependencies(self):
  c=p.recipe(Path('/CPU_ONLY_UNCREATED/pti-build'));self.assertNotIn('--device',c);self.assertNotIn('--gpus',c);self.assertIn(p.IMAGE,c);self.assertIn(str(p.SOURCE)+':/src:ro',c);self.assertIn('BUILD_WITH_L0=1',c[-1]);self.assertIn('BUILD_WITH_XPTI=0',c[-1]);self.assertIn('BUILD_WITH_PERFETTO=OFF',c[-1]);self.assertIn(p.LEVEL_ZERO,c[-1]);self.assertIn(p.COMPUTE_RUNTIME,c[-1]);self.assertIn('--target unitrace unitrace_tool',c[-1]);self.assertNotIn('ctest',c[-1])
 def test_original_helper_not_part_of_build_mounts(self):
  c=p.recipe(Path('/CPU_ONLY_UNCREATED/pti-build'));mounts=[c[i+1]for i,x in enumerate(c)if x=='-v'];self.assertEqual(len(mounts),2);self.assertFalse(any('/model:'in m or '/inputs:'in m or '/dev/dri'in m for m in mounts))
 def test_invalid_output_refused(self):
  self.assertRaises(ValueError,p.recipe,Path('relative'));self.assertRaises(ValueError,p.recipe,Path('/tmp'))
if __name__=='__main__':unittest.main()
