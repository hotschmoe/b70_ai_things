"""NEW explicit buffered plan/producer metadata gates, external boundaries mocked."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_api_cache_positive_buffered_v4 as q
import run_batch_api_cache_positive_buffered_v4 as runner
from audit_batch_numerical_suite_v7 import snapshot_plan_join
class Controls(unittest.TestCase):
 def test_prepare_visible_original_sourceview_and_exact_recipe(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'prepared';original={'driver_sha256':q.sha(Path(q.base.__file__)),'args':['CPU_EXACT_NN_ARGS'],'env':{'CPU_FIXED':'1'},'API_driver_sha256':q.sha(Path(q.base.api.__file__))}
   def prepare(a):out.mkdir();q.write(out/'plan.json',original)
   with patch.object(q.semantic,'prepare',prepare),patch.object(q.semantic,'admit',return_value={'CPU_OTHER_GATES':True}):q.prepare(SimpleNamespace(output=out));plan=q.read(out/'plan.json');self.assertEqual(plan['positive_V2_source_preparation'],original);self.assertEqual(plan['args'],original['args']);self.assertEqual(plan['env'],original['env']);self.assertEqual(plan['buffered_trace_generation'],3);self.assertFalse(plan['matched_buffer_IO_savings_qualified']);q.manifest_binding(plan);bad=copy.deepcopy(plan);bad['args']=['CHANGED_MATH']
   with patch.object(q.semantic,'admit',return_value={}):
    with self.assertRaises(ValueError):q.manifest_binding(bad)
 def test_real_snapshotSHA_before_completed_bufferedreport(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'child';out.mkdir();plan={'CPU_SYNTHETIC':True};external=root/'plan.json'
   for path in (external,root/'input-plan.snapshot.json',out/'plan.snapshot.json'):q.write(path,plan)
   with patch.object(runner,'artifact_bindings',return_value={'CPU_ONLY':True}):child=runner.seal_report(plan,out,{'collection_and_teardown_passed':True})
   self.assertEqual(child['buffered_trace_generation'],3);self.assertEqual(child['terminal_association_generation'],2);self.assertFalse(child['actual_cached_state_handoff_qualified']);snapshot_plan_join(root,{'plan':str(external),'plan_sha256':q.sha(external)},plan,child)
 def test_unchanged_warm_multirow_gate_still_required(self):
  source=Path(runner.__file__).read_text();self.assertIn("require(warm_prefixes['actual_completed_multirow_events']>0",source);self.assertIn("request_policy=plan['API_warm_request_policy']",source);self.assertIn("request_policy=plan['API_target_request_policy']",source);self.assertIn('/controller/batch_api_trace_v3.py',source);self.assertIn("trace_status['passed'] is True",source)
 def test_current_semantic_registry_gate_keeps_original_wholepredicate_unclaimed(self):
  import re
  import positive_semantic_registry_admission_v4 as current
  alias=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',current.registry.POSITIVE12.read_text(),re.M)[0];binding=current.registry_gate(alias);self.assertEqual(binding['sha256'],'86621c71b829c75cc2f7312248932e9a8bb276018db3c0bef77f1fdb58678052');self.assertFalse(binding['semantic_entry_preservation']['original_current_global_gates_passed']);self.assertIn('NOT claimed',binding['scope'])
if __name__=='__main__':unittest.main()
