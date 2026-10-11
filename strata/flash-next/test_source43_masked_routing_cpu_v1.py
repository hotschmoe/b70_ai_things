"""Source-only generation routing and closure controls; no SDK payload reads."""
import ast,json,unittest
from pathlib import Path
import qualify_indexer_materialized43_masked_v2 as q
class Controls(unittest.TestCase):
 def test_all_derived_expensive_routes_are_explicit43(self):
  ns=q.derived()
  for name in ('main','finalized_binding'):
   nodes=ast.parse(Path(q.__file__).read_text());self.assertTrue(callable(ns[name]))
  source=Path(q.__file__).read_text()
  self.assertIn('indexer_materialized43_masked_inputs_v2 import producer_binding',source)
  self.assertIn('indexer_materialized43_masked_control_v1 import native_trace',source)
  self.assertIn('actual_fresh43_upload_binding',source)
  self.assertNotIn('from admit_source_upload41',source)
  self.assertEqual(q.SDK.name,'strata-native-hc-engine-20261011T120045Z-kfiohqmn')
 def test_genuine70_8_and8archive_expected_roster(self):
  plan=json.loads(q.PLAN.read_text());expected=plan['actual_consumed_source_bindings']
  self.assertEqual(len([p for p in expected if '/source/'in p]),70)
  self.assertEqual(len([p for p in expected if p.endswith('.a')]),8)
  self.assertEqual(len(plan['required_fresh_binary_outputs']),8)
  self.assertTrue(all('120045Z-kfiohqmn'in p for p in expected if p.endswith('.a')))
  self.assertFalse(plan['model_baseline_or_constructor_proof_transferred'])
 def test_private_namespace_runtime_entry_is_truthful43(self):
  ns=q.derived();ns['sha']=lambda _: 'CPU_SYNTHETIC_SHA'
  argv,compile,runtime=ns['expected_recipes']('/CPU_SYNTHETIC/out','/CPU_SYNTHETIC/fixture',123)
  self.assertIn('/leaf/indexer_materialized43_masked_gpu_v1.cpp',argv)
  self.assertIn('idx43mode0',runtime[-1]);self.assertIn('idx43mode1',runtime[-1])
  self.assertNotIn('idx41',str(runtime));self.assertNotIn('half37.after',str(runtime))
if __name__=='__main__':unittest.main()
