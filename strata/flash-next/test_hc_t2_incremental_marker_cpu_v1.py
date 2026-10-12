import copy,unittest
from pathlib import Path
import build_hc_t2_paired_mtp_v4 as b
class Controls(unittest.TestCase):
 def test_actual_recipe_equivalence_and_only_verify_compile(self):
  old=b.read(b.PARENT/'compile-recipe.json');new=copy.deepcopy(old);new['marked_source_plan_sha256']=b.MARKED_SHA;new['mounts']['overlay_RO']='new/overlay';new['mounts']['output_RW']='new/output';new['overlay_sha256']['sycl/src/core/verify.cpp']='f'*64
  self.assertTrue(b.unchanged_recipe(old,new));bad=copy.deepcopy(new);bad['compile_commands'][1]['argv'].append('-ffast-math')
  with self.assertRaises(ValueError):b.unchanged_recipe(old,bad)
  bad=copy.deepcopy(new);bad['overlay_sha256']['sycl/include/strata/core/hc_t2_route_proof.hpp']='e'*64
  with self.assertRaises(ValueError):b.unchanged_recipe(old,bad)
  script=b.shell(new);self.assertEqual(script.count(' -c '),1);self.assertIn('/overlay/sycl/src/core/verify.cpp',script);self.assertIn('cp /parent-output/main.hc-t2.o',script);self.assertNotIn('ar d ',script)
if __name__=='__main__':unittest.main()
