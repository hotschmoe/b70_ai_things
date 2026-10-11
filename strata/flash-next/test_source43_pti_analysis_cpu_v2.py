import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
import analyze_source43_pti_latency_v2 as a
import test_source43_pti_analysis_cpu_v1 as old
class Controls(old.Controls):
 def setUp(self):self.p=patch.object(old,'a',a);self.p.start()
 def tearDown(self):self.p.stop()
class NewControls(unittest.TestCase):
 def test_actual_actor3_PLAN_wrong_generation_source_refusal(self):
  import run_source43_pti_latency_v3 as actor
  expected=a.host.bound(actor.PLAN)['sha256'];report={'source_plan_sha256':expected};admission={'source_plan_sha256':expected,'source_generation':43,'materialized_mode':0};self.assertEqual(a.generation_gate(report,admission),expected)
  for key,value in(('source_generation',41),('materialized_mode',False),('source_plan_sha256','old')):
   wrong={**admission,key:value}
   with self.assertRaises(ValueError):a.generation_gate(report,wrong)
 def test_raw_tick_union_function_AST_unchanged(self):
  current=next(n for n in ast.parse(Path(a.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='raw_device_metrics');original=next(n for n in ast.parse(Path(a.__file__).with_name('analyze_source43_pti_latency_v1.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='raw_device_metrics');self.assertEqual(ast.dump(current),ast.dump(original))
 def test_strict_current_census_source_callback_used_without_model_admission(self):
  text=Path(a.__file__).read_text();self.assertIn('from validate_source43_pti_latency_v3 import census_binding',text);self.assertIn("admission['mapped_injection']",text);self.assertNotIn('from validate_source43_pti_latency_v3 import finalized_binding',text);self.assertIn("'current_model_or_health_readmission_performed':False",text);self.assertIn("'bandwidth_qualified':False",text)
if __name__=='__main__':unittest.main()
