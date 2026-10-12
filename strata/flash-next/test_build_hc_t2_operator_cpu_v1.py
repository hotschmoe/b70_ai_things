import unittest
from pathlib import Path
import build_hc_t2_operator_v1 as b
class Controls(unittest.TestCase):
 def test_genuine_recipe_and_archive_filter(self):
  path=Path('/mnt/vm_8tb/b70/build/hc-t2-operator-compile-inputs-v1/compile-recipe.json');self.assertEqual(b.sha(path),b.RECIPE_SHA);recipe=b.read(path);script=b.shell(recipe)
  self.assertIn('test ! -e /dev/dri',script);self.assertIn('cmp /sdk/build/packages.txt',script)
  self.assertEqual(len(recipe['commands']),4);self.assertLess(recipe['link_argv'].index('/out/hc_projection.t2.o'),recipe['link_argv'].index('/out/libstrata_kernels.without_hc.a'))
  names=recipe['archive_filter']['remove_members'];before=['a.o',names[0],names[1],'b.o'];self.assertTrue(b.archive_members(before,['a.o','b.o'],names))
  with self.assertRaises(ValueError):b.archive_members(before+[names[0]],['a.o','b.o'],names)
  with self.assertRaises(ValueError):b.archive_members(before,['a.o'],names)
  self.assertFalse(b.absent(b'permission denied','owned'))
  self.assertTrue(b.absent(b'Error: No such object: owned','owned'))
if __name__=='__main__':unittest.main()
