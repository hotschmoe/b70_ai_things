"""Readonly successor must retain actual actor3 and original native binding."""
import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import validate_source43_num10_zero_v4 as fixed
import validate_source43_num10_zero_v3 as old
class Controls(unittest.TestCase):
 def test_exact_reader_extra_collection_change_and_truthful_actor3(self):
  a=Path(old.__file__).read_text();b=Path(fixed.__file__).read_text();tree=ast.parse(b)
  self.assertIs(fixed.a,old.a);self.assertTrue(fixed.a.__file__.endswith('run_source43_num10_zero_v3.py'));self.assertEqual(fixed.a.PLAN.name,'source43-num10-model-source-plan-v3.json')
  node_old=next(n for n in ast.parse(a).body if isinstance(n,ast.FunctionDef)and n.name=='derived');node_new=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='derived')
  for n in ast.walk(node_new):
   if isinstance(n,ast.ImportFrom)and n.module=='source43_num10_numeric_v2':n.module='source43_num10_numeric_v1'
  self.assertEqual(ast.dump(node_old,include_attributes=False),ast.dump(node_new,include_attributes=False))
 def test_existing_actor_binding_is_not_relabelled(self):
  base={'source_plan_sha256':'777-ORIGINAL-ACTOR','report_sha256':'ORIGINAL-REPORT'}
  with patch.object(fixed,'source_binding',return_value='NEW-READER-ONLY'),patch.object(fixed,'derived',return_value={'finalized_binding':lambda root:dict(base)}):
   result=fixed.finalized_binding('/CPU_SYNTHETIC')
  self.assertEqual(result['source_plan_sha256'],base['source_plan_sha256']);self.assertEqual(result['report_sha256'],base['report_sha256']);self.assertEqual(result['readonly_reader_fix_source_plan_sha256'],'NEW-READER-ONLY')
 def test_reader_source_postcheck_on_original_failure(self):
  calls=[]
  def check():calls.append(1);return'CPU_FIX_SOURCE'
  def fail(root):raise RuntimeError('original current gate failed')
  with patch.object(fixed,'source_binding',side_effect=check),patch.object(fixed,'derived',return_value={'finalized_binding':fail}):
   with self.assertRaisesRegex(RuntimeError,'original current gate failed'):fixed.finalized_binding('/CPU_SYNTHETIC')
  self.assertEqual(calls,[1,1])
if __name__=='__main__':unittest.main()
