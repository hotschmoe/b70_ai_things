"""Public raw-read closure after original admission, including failed reads."""
import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import validate_source43_num10_zero_v2 as r
class Controls(unittest.TestCase):
 def test_private_namespace_uses_true_new_producer_and_intake(self):
  n=r.derived();self.assertIs(n['a'],r.a);self.assertEqual(r.a.PLAN.name,'source43-num10-model-source-plan-v2.json');self.assertTrue(r.a.intake.__file__.endswith('_v2.py'))
 def run_fixture(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'report.json').write_text('{}');(root/'capture').write_bytes(b'original');calls=[]
   def tree(p):return {f.name:r.c.sha(f)for f in p.iterdir()if f.name!='report.json'}
   def collect(p):
    calls.append('collect')
    if kind=='changed':(p/'capture').write_bytes(b'changed')
    if kind=='failure':raise RuntimeError('original raw failure')
    return {'actual_result':True}
   node=next(n for n in ast.walk(ast.parse(Path(r.__file__).read_text()))if isinstance(n,ast.FunctionDef)and n.name=='with_actual_raw');ns={**vars(r),'namespace':{'tree':tree},'collect_actual':collect}
   exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[CPU-actual-raw-close]','exec'),ns)
   with patch.object(r.a,'closure',side_effect=lambda:calls.append('source')or'CPU_SYNTHETIC_SOURCE'):
    if kind=='valid':self.assertTrue(ns['with_actual_raw'](root)['actual_result'])
    elif kind=='changed':
     with self.assertRaises(ValueError):ns['with_actual_raw'](root)
    else:
     with self.assertRaisesRegex(RuntimeError,'original raw failure'):ns['with_actual_raw'](root)
   self.assertEqual(calls,['source','collect','source']if kind!='changed'else['source','collect'])
 def test_original_result_preserved(self):self.run_fixture('valid')
 def test_consistent_tensor_replacement_rejected(self):self.run_fixture('changed')
 def test_original_failure_still_rechecks_source_and_roster(self):self.run_fixture('failure')
 def test_parent_only_namespace_diff_preserves_recipe_and_lifecycle(self):
  here=Path(r.__file__).parent;one=(here/'run_source43_num10_zero_v1.py').read_text();two=(here/'run_source43_num10_zero_v2.py').read_text();self.assertEqual(one,two.replace('source43_num10_intake_v2','source43_num10_intake_v1').replace('source43-num10-model-source-plan-v2.json','source43-num10-model-source-plan-v1.json'))
if __name__=='__main__':unittest.main()
