import ast,unittest
from pathlib import Path
import test_full_cache_shared_suite_cpu_v3 as fixture
from full_cache_shared_phase_contract_v2 import scenario_schedule
import live_cache_identity_controller_v2 as c
import run_live_cache_identity_actor_v2 as a

class Actor(unittest.TestCase):
 def test_exact_five_native_requests_three_complete_raw49_owners(self):
  phases,roster,selected=c.family_schedule({'schedule':scenario_schedule(fixture.Suite().case(),'shared')['phases']});self.assertEqual([(p['name'],p['expected_actual_RID_set'])for p in phases],[('warm',[1,2]),('prime0',[3]),('prime1',[4]),('repeat_original',[5])]);self.assertEqual(selected['selected_actual_RIDs'],[3,4,5]);self.assertEqual(phases[2]['expected_reused'],[272]);self.assertEqual(phases[3]['expected_reused'],[272]);self.assertEqual(phases[2]['rows'],phases[3]['rows'])
 def test_concrete_actor_retains_original_model_methods_and_probes_after_prime(self):
  source=a.adapted_source();ast.parse(source);self.assertIn('/controller/full_cache_live_identity_api_v2.py',source);self.assertIn("phase['name']=='prime1'",source);self.assertIn('loaded_probes(url,',source);self.assertIn('retain_until_settled',source);self.assertNotIn('actual_model_weight_data_changed=True',source)
 def test_current_source_ledger_required_before_runtime(self):
  self.assertEqual(c.source_binding()['source_plan_sha256'],c.sha(c.SOURCE_PLAN));self.assertIn('child_wait(plan,manifest,args.output)',Path(c.__file__).read_text());self.assertIn('post_ack_seal(plan,sys.modules[__name__]',Path(c.__file__).read_text())
