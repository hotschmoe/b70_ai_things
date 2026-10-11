import ast,unittest
from pathlib import Path
from unittest.mock import patch
import analyze_source43_warm_pti_v3 as a
class Tests(unittest.TestCase):
 def test_current_generation_and_runtime_dispatch(self):
  expected=a.host.bound(a.warm_actor.PLAN)['sha256'];report={'source_plan_sha256':expected};admit={'source_plan_sha256':expected,'source_generation':43,'materialized_mode':0};self.assertEqual(a.generation_gate(report,admit),expected)
  with self.assertRaises(ValueError):a.generation_gate(report,{**admit,'source_plan_sha256':'v2'})
  runtime={'injection':{'sha256':a.warm_actor.fresh_tool.LIB_SHA},'tool':{'sha256':a.warm_actor.fresh_tool.TOOL_SHA},'source_plan_sha256':a.warm_actor.fresh_tool.SOURCE_SHA,'paused_regular_list_retention_required':True,'intake_path':'/original/report.json','exact_intake_sha256':'a'*64,'full_model_coverage_qualified':False,'clock_alignment_qualified':False}
  with patch.object(a.warm_actor.fresh_tool,'binding',return_value=runtime)as binding:
   self.assertIs(a.runtime_join(runtime),runtime);binding.assert_called_once_with('/original/report.json','a'*64)
  with patch.object(a.warm_actor.fresh_tool,'binding')as binding:
   with self.assertRaises(ValueError):a.runtime_join({**runtime,'injection':{'sha256':'old'}})
   binding.assert_not_called()
 def test_unchanged_interval_math_and_actual_names(self):
  def node(file,name):return next(n for n in ast.parse(Path(file).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==name)
  for name in('raw_device_metrics','selected_host_metric'):
   self.assertEqual(ast.dump(node(a.__file__,name)),ast.dump(node(Path(a.__file__).with_name('analyze_source43_warm_pti_v2.py'),name)))
  self.assertEqual(a.kernel_counts([{'instance_id':1,'kernel_name':'compute'},{'instance_id':2,'kernel_name':'compute'},{'instance_id':3,'kernel_name':'fill'}]),[{'kernel_name':'compute','instances':2},{'kernel_name':'fill','instances':1}])
  with self.assertRaises(ValueError):a.kernel_counts([{'instance_id':True,'kernel_name':'compute'}])
if __name__=='__main__':unittest.main()
