"""Producer-shaped source integration, exact current171 and unchanged fullsuite."""
import ast,copy,hashlib,json,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_baseline_admission_v12 as baseline
class Controls(unittest.TestCase):
 def test_fresh171_rejects_historical169_and_foreign_entry(self):
  from registry_c140_shared_association_v3 import C140
  from strict_registry_yaml_v3 import parse
  from registry_c140_chunk173_association_v1 import CURRENT_SHA
  prepared={'registry_sha256':CURRENT_SHA,'alias':parse(C140.read_bytes())[0]['served_model_id']};self.assertEqual(baseline.registry_binding(prepared)['actual_prepared_models'],173)
  import full_cache_shared_baseline_admission_v7 as old
  self.assertRaises(ValueError,old.registry_binding,prepared)
  for key,value in [('registry_sha256','0'*64),('alias','foreign')]:
   bad=dict(prepared);bad[key]=value;self.assertRaises(ValueError,baseline.registry_binding,bad)
 def test_every_producer_uses_owned_waiter_and_incremental_phase_metadata(self):
  here=Path(__file__).parent;producer=(here/'full_cache_shared_api_trace_v12.py').read_text();actor=(here/'run_full_cache_shared_runtime_v12.py').read_text();reader=(here/'validate_full_cache_shared_runtime_v12.py').read_text()
  for text in(producer,actor,reader):ast.parse(text)
  self.assertIn('waiter_install(server,emit,lambda engine:',producer);self.assertIn("waiter_observer.proof(packet['calls'])",producer);self.assertIn('trace_anchor=sink.anchor(marker)',producer);self.assertIn('all_sent_BSTOPs_observed(current)',actor);self.assertIn("limit=time.monotonic()+60",actor);self.assertIn('all_events,last,trace,bounds,stream_snapshot=stream.freeze()',actor);self.assertIn('snapshot_binding(',reader);self.assertIn('adjudicate(events,proof',reader);self.assertIn('actual_original_C1401403_proof',(here/'full_cache_shared_baseline_admission_v12.py').read_text())
 def test_fullscope_declaration_unchanged_and_no_native_math_policy(self):
  import full_cache_shared_suite_v11 as old,full_cache_shared_suite_v12 as new
  self.assertEqual(ast.dump(ast.parse(Path(old.__file__).read_text()).body[0]),ast.dump(ast.parse(Path(new.__file__).read_text()).body[0]))
  import test_full_cache_shared_suite_cpu_v12 as fixture
  case=fixture.Suite().case();self.assertEqual(old.declaration(case),new.declaration(case));self.assertEqual(len(new.declaration(case)['actors']),9);self.assertTrue(new.declaration(case)['serial_persisted_family_required'])
if __name__=='__main__':unittest.main()
