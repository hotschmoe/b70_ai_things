import unittest,json
from pathlib import Path
import build_original_mtp53_two_stage_v6 as b
import run_original_mtp53_two_stage_v5 as r
class Controls(unittest.TestCase):
 def test_actual_recipe_and_completed_normal_cuts(self):
  recipe=b.read(Path('/mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v6/compile-recipe.json'));script=b.shell(recipe)
  self.assertIn('test ! -e /dev/dri',script);self.assertIn('cmp /sdk/build/compiler-version.txt',script);self.assertEqual(recipe['two_stage_source_plan_sha256'],b.PLAN_SHA)
  prompt=json.loads(r.pp_policy.REFERENCE.read_bytes())['requests'][0]['ids'];ends=[24,1536,2041,2047];text='';chunks=[];start=0
  for end in ends:
   text+='MTP53_PAIR_PREFILL_CUT begin=%d end=%d normal_reference_policy=1 actual_cache_enabled=0\n'%(start,end)
   pos=start
   while pos<end:
    take=min(512,end-pos);chunks.append({'begin':str(pos),'end':str(pos+take),'rows':str(take)});pos+=take
   start=end
  self.assertEqual(r.prefix_rows(text,chunks,prompt)['logical_end_positions'],ends)
  with self.assertRaises(ValueError):r.prefix_rows(text,chunks[:-1],prompt)
  with self.assertRaises(ValueError):r.prefix_rows(text.replace('end=24 ','end=25 ',1),chunks,prompt)
if __name__=='__main__':unittest.main()
