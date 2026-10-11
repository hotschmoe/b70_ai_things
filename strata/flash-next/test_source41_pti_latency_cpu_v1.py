import json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import source41_pti_raw_budget_v1 as b
import source41_pti_runtime_v1 as runtime
import run_source41_pti_latency_v1 as actor
import validate_source41_pti_latency_v1 as reader
import collect_source41_model_pti_v1 as collector
import test_source41_model_actor_cpu_v3 as fixtures
class Controls(unittest.TestCase):
 def test_unfilled_runtime_refuses_before_lease_or_model(self):
  with self.assertRaises(ValueError):runtime.binding(Path(runtime.__file__).with_name('source41_pti_runtime_intake_v1.json'))
 def test_actual_raw_budget_overshoot_latches_and_retires(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);notice=threading.Event();watch=b.Watcher(root,notice.set,cap=32,interval=.01);watch.start();(root/'raw.json').write_bytes(b'x'*40);self.assertTrue(notice.wait(1));proof=watch.close();self.assertFalse(proof['passed']);self.assertTrue(proof['thread_retired']);self.assertEqual(proof['observed_overshoot_bytes'],8);self.assertGreaterEqual(proof['peak_observed_bytes'],40)
 def test_raw_budget_normal_and_symlink_refusal(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'raw').write_bytes(b'123');watch=b.Watcher(root,lambda:None,cap=32,interval=.01);watch.start();self.assertTrue(watch.close()['passed']);(root/'alias').symlink_to(root/'raw')
   with self.assertRaises(ValueError):b.measure(root)
 def test_both_H36_ON_only_PTI_flag_wrapper_delta(self):
  m,cfg,inputs=fixtures.Controls().fixture();runtime_binding={'image':m['runtime']['image'],'install_root':'/fresh/install'}
  with patch.object(actor,'_RUNTIME',runtime_binding),patch.dict('os.environ',{},clear=True):
   left=actor.phase_recipe(actor.PHASES[0],m,cfg,inputs,'a'*64)[1];right=actor.phase_recipe(actor.PHASES[1],m,cfg,inputs,'a'*64)[1]
   self.assertEqual(left['requests'],right['requests']);self.assertEqual(right['native_argv'],['/pti/bin/unitrace',*runtime.OPTIONS,*left['native_argv']]);self.assertEqual(left['native_env']['STRATA_CRITICAL_PATH_TRACE'],'1');self.assertEqual(right['native_env']['STRATA_CRITICAL_PATH_TRACE'],'1');self.assertEqual({k:v for k,v in right['native_env'].items()if left['native_env'].get(k)!=v},runtime.CLOCK_ENV);self.assertEqual(right['total_raw_cap_bytes'],1<<30);self.assertEqual(right['metadata_cap_bytes'],512<<20)
 def test_inherited_metrics_preload_or_clock_refused(self):
  for key in('UNITRACE_KernelMetrics','UNITRACE_ChromeKmdLogging','UNITRACE_CLOCK_METADATA_PATH','LD_PRELOAD'):
   with patch.dict('os.environ',{},clear=True),self.assertRaises(ValueError):runtime.inherited_env({key:'0'})
 def test_own_reader_native_namespace_and_lifecycle_routing(self):
  self.assertIs(reader._PORTS['a'],actor);self.assertIs(actor._PORTS['Protocol'],actor.Protocol);self.assertIs(actor._PORTS['life'],actor.life);self.assertIs(actor._PORTS['set_runtime'],actor.set_runtime)
 def test_model_clock_collector_no_fixture_cardinality_requirement(self):
  collect=collector.clock_port({});self.assertTrue(callable(collect));self.assertIn('len(instances)',__import__('inspect').getsource(collector.clock_port));self.assertNotIn('==126',Path(collector.__file__).read_text());self.assertNotIn('==631',Path(collector.__file__).read_text())
if __name__=='__main__':unittest.main()
