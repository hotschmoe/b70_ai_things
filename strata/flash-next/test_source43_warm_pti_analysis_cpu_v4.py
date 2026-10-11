import ast,json,tempfile,unittest,hashlib
from pathlib import Path
import analyze_source43_warm_pti_v4 as a
class Tests(unittest.TestCase):
 def test_historical_component_keeps_failed_parent(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);raw=json.dumps({'passed':False,'errors':['original failure']}).encode();(root/'report.json').write_bytes(raw)
   component={'root':str(root),'report_sha256':hashlib.sha256(raw).hexdigest(),'passed':True,'historical_metadata_only':True,'original_parent_passed':False,'whole_parent_qualified':False,'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'fresh_public_full4_rehashed':False,'component_original_recorded_phase_gates_passed':True,'phase':'source43-warm-pti-OFF','source_generation':43,'materialized_mode':0,'original_parent_errors':['original failure']};parent={'OFF_component_prerequisite':component};admit=dict(parent)
   actual,report=a.component_join(component,parent,admit);self.assertEqual(actual,root);self.assertIs(report['passed'],False)
   with self.assertRaises(ValueError):a.component_join({**component,'original_parent_passed':True},parent,admit)
   (root/'report.json').write_text(json.dumps({'passed':False,'errors':['changed']}))
   with self.assertRaises(ValueError):a.component_join(component,parent,admit)
 def test_v4_route_and_unchanged_interval_math(self):
  expected=a.host.bound(a.warm_actor.PLAN)['sha256'];self.assertEqual(a.generation_gate({'source_plan_sha256':expected},{'source_plan_sha256':expected,'source_generation':43,'materialized_mode':0}),expected)
  self.assertEqual(a.PHASES,('source43-warm-pti-ON',))
  def node(file,name):return next(n for n in ast.parse(Path(file).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==name)
  for name in('raw_device_metrics','selected_host_metric','kernel_counts'):
   self.assertEqual(ast.dump(node(a.__file__,name)),ast.dump(node(Path(a.__file__).with_name('analyze_source43_warm_pti_v3.py'),name)))
if __name__=='__main__':unittest.main()
