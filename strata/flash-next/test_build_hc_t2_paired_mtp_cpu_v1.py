import unittest
from pathlib import Path
import build_hc_t2_paired_mtp_v1 as b
class Controls(unittest.TestCase):
 def test_actual_recipe_and_two_filters(self):
  p=Path('/mnt/vm_8tb/b70/build/hc-t2-paired-mtp-compile-inputs-v1/compile-recipe.json');self.assertEqual(b.sha(p),b.RECIPE_SHA);r=b.read(p);s=b.shell(r)
  self.assertEqual(r['hc_source_plan_sha256'],b.PLAN_SHA);self.assertEqual(r['two_stage_source_plan_sha256'],b.MTP_SHA);self.assertIn('test ! -e /dev/dri',s)
  for i,f in enumerate(r['archive_filters']):
   members=f['remove_members'];old=['old-first.o']+members+['old-last.o'];self.assertTrue(b.archive_members(old,['old-first.o','old-last.o'],members))
   with self.assertRaises(ValueError):b.archive_members(old+[members[0]],['old-first.o','old-last.o'],members)
  link=r['link_argv'];self.assertTrue(all(link.index('/out/'+v['name'])<link.index('-Wl,--start-group')for v in r['compile_commands']))
if __name__=='__main__':unittest.main()
