import unittest
from pathlib import Path
import run_original_mtp53_two_stage_v4 as r
class Controls(unittest.TestCase):
 def test_fresh_marker_and_actual_compile_metadata(self):
  root=Path('/mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v5');recipe=r.read(root/'compile-recipe.json');receipt=r.read(root/'output/receipt.json')
  self.assertEqual(recipe['two_stage_source_plan_sha256'],r.sha(r.HERE/'original-publisher-mtp53-two-stage-proposal-v5.json'));self.assertEqual(receipt['recipe_sha256'],r.sha(root/'compile-recipe.json'));self.assertEqual(receipt['outputs']['original-mtp53-two-stage'],'1316eceb3a2d901d6811855779307a5cd0d7907530e2dde0815e439c24220963')
  marker='MTP53_PAIR_FRESH_RESET stages=2 devices=0,1 ranges=0:24,24:48 native52_public_reset=1 before_target_bind=1\n'
  self.assertTrue(r.fresh_reset_marker(marker))
  for bad in('',marker*2,marker.replace('stages=2','stages=1')):
   with self.assertRaises(ValueError):r.fresh_reset_marker(bad)
  cfg=r.read(r.BASE)
  for mode,ids in(('controls',[248045]),('off',[248045]*2048),('on',[248045]*2048)):
   argv=r.native_options(cfg,mode,ids);self.assertEqual(argv[argv.index('--prefill')+1],'512');self.assertEqual(argv[argv.index('--prompt-cache')+1],'0')
  self.assertEqual(r.selection(0,1),([0],1))
if __name__=='__main__':unittest.main()
