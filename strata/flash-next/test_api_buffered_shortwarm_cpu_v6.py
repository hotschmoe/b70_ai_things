"""NEW input-admission/source controls, no Docker/GPU/helper/model execution."""
import copy,ast,inspect,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_api_cache_positive_buffered_v6 as c
import api_shortwarm_positive_contract_v6 as contract
import qualify_batch_api_cache_positive_buffered_v6 as parent
import run_batch_api_cache_positive_buffered_v6 as runner
import produce_api_shortwarm_tokenizer_fixture_v1 as producer
class Tests(unittest.TestCase):
 def case(self):return c.generator.case_from_fixture(contract.FIXTURE_ROOT,18339)
 def test_authentic_fixture(self):
  f,b=contract.fixture_binding();self.assertEqual([len(r['ids']) for r in f['fixtures']['warm']],[47,47]);self.assertEqual([len(r['ids']) for r in f['fixtures']['target']],[235,235]);self.assertFalse(b['warm_two_row_runtime_qualified'])
 def test_exact_case(self):c.case_admission(self.case())
 def test_wrong_case_fields(self):
  for key,value in [('shortwarm_case_generation',0),('matched_buffer_only_A_B_input_equivalent',True),('api_warm_max_new_by_request',[64,64]),('genuine_runtime_preparable',True)]:
   case=self.case();case[key]=value
   with self.assertRaises(ValueError):c.case_admission(case)
 def test_old_case_before_expensive(self):
  with patch.object(c,'source_gate'),patch.object(c,'read',return_value={"port":18339}),patch.object(c.original.base,'prepare') as base,patch.object(c.c1,'leased') as leased:
   with self.assertRaises(ValueError):c.prepare(SimpleNamespace(spec=Path('/unused'),kind='api',lane='source35'))
   base.assert_not_called();leased.assert_not_called()
 def test_prepare_new_fields_and_restores_frozen_hooks(self):
  case=self.case();ns=c.namespace();ns['read']=lambda p:copy.deepcopy(case);proxy=SimpleNamespace(**vars(contract));proxy.profile=lambda candidate:{'fixture':case['authentic_fixture_binding']};proxy.registry_gate=lambda alias:{'CPU':'ONLY'};ns['contract']=proxy;ns['recipe']=lambda *a:([],{});ns['api']=SimpleNamespace(experimental_alias=lambda *a:'CPU_ALIAS',set_arg=lambda *a:None,__file__=runner.__file__);captured=[]
  before=(c.original.base.api,c.original.base.manifest_binding,c.original.base.read)
  def fake_prepare(a):
   plan={};c.original.base.manifest_binding(plan);captured.append(plan)
  with patch.object(c,'namespace',return_value=ns),patch.object(c.original.base,'prepare',side_effect=fake_prepare),patch.object(c,'manifest_binding',return_value={'CPU':'ONLY'}),patch.object(c,'read',return_value=case),patch.object(c,'source_gate'):
   c.prepare(SimpleNamespace(spec=Path('/unused'),prepared=Path('/unused'),diagnostic=1,kind='api',lane='source35'))
  self.assertEqual((c.original.base.api,c.original.base.manifest_binding,c.original.base.read),before);plan=captured[0];self.assertEqual(plan['shortwarm_case_generation'],1);self.assertEqual(plan['api_warm_max_new_by_request'],[32,32]);self.assertEqual(plan['authentic_positive_case'],case);self.assertEqual(plan['buffered_wrapper_generation'],6);self.assertEqual(plan['API_driver_sha256'],c.sha(Path(ns['api'].__file__)) if hasattr(ns['api'],'__file__') else c.sha(Path(runner.__file__)))
 def test_manifest_wrong_generation_before_model(self):
  with patch.object(c.original,'FROZEN_MANIFEST') as baseline:
   with self.assertRaises(ValueError):c.manifest_binding({'buffered_wrapper_generation':5})
   baseline.assert_not_called()
 def test_parent_exact_chronology(self):
  source=parent.adapted_source();ast.parse(source);self.assertIn('import batch_api_cache_positive_buffered_v6 as ctrl',source);self.assertIn("parent['child_launch_started_epoch']=time.time()",source);self.assertIn("'started':identity['started']",source)
 def test_real_warm_two_row_not_waived(self):
  source=Path(runner.__file__).read_text();self.assertIn("warm_prefixes['actual_completed_multirow_events']>0",source);self.assertIn('from api_owned_terminal_association_v2 import associate_events',source);self.assertIn('max_new=32',source)
 def test_seal_metadata(self):
  source=inspect.getsource(runner.seal_report);self.assertIn('buffered_wrapper_generation=6',source);self.assertIn("plan_sha256=sha(out/'plan.snapshot.json')",source);self.assertIn('shortwarm_case_generation=1',source)
 def test_namespace_manifest_source(self):
  ns=c.namespace();self.assertEqual(ns['SOURCE_PLAN'],c.SOURCE_PLAN);self.assertEqual(ns['contract'],contract);self.assertEqual(ns['api'],runner)
if __name__=='__main__':unittest.main()
