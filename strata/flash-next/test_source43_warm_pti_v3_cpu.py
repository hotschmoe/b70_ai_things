import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
import run_source43_warm_pti_v3 as actor
import source43_paused_pti_runtime_v1 as runtime
import validate_pti_paused_regular_fixture_v2 as fixture

class Controls(unittest.TestCase):
 def test_actual_compiled_parent_routes_history(self):
  self.assertIs(actor._PORTS['prerequisite'],actor.prerequisite)
  with patch.object(actor.historical,'prerequisite',return_value={'historical_completed_evidence_only':True})as call:
   self.assertTrue(actor._PORTS['prerequisite'](Path('/tmp/history'))['historical_completed_evidence_only']);call.assert_called_once()
 def test_current_admission_unchanged_and_failure_propagates(self):
  with patch.object(actor.base,'admission',return_value=('metadata','config'))as call:
   self.assertEqual(actor.admission({'current':True}),('metadata','config'));call.assert_called_once_with({'current':True})
  with patch.object(actor.base,'admission',side_effect=ValueError('current identity failed')):
   with self.assertRaisesRegex(ValueError,'current identity'):actor.admission({})
 def envelope(self):
  return {'intake_path':'/tmp/fixture/report.json','exact_intake_sha256':'a'*64,'tool':{'sha256':runtime.TOOL_SHA},'injection':{'sha256':runtime.LIB_SHA},'install_root':str(runtime.BUILD/'install'),'source_plan_sha256':runtime.SOURCE_SHA,'old_dependency_receipt_transferred':False,'paused_regular_list_retention_required':True,'query_sample_every':512,'full_model_coverage_qualified':False,'clock_alignment_qualified':False}
 def proof(self):return {'fixture_success_revalidated':True,'actual_retention_OFF_ON_discriminated':True,'runtime_envelope':self.envelope()}
 def test_current_census_mount(self):
  self.assertEqual(actor.census.EXPECTED_HOST_ROOT,str(runtime.BUILD/'install'))
  self.assertIn('paused-regular',actor.census.EXPECTED_BIND_ROOT)
 def test_fresh_fixture_runtime_dispatch(self):
  with patch.object(fixture,'finalized_binding',return_value=self.proof())as call:
   result=runtime.binding(Path('/tmp/fixture/report.json'),'a'*64);self.assertEqual(result,self.envelope());call.assert_called_once_with(Path('/tmp/fixture'),'a'*64)
  self.assertEqual(runtime.CLOCK_ENV['UNITRACE_RETAIN_PAUSED_REGULAR_LISTS'],'1')
 def test_failed_fixture_and_old_injection_refused(self):
  proof=self.proof();proof['fixture_success_revalidated']=False
  with patch.object(fixture,'finalized_binding',return_value=proof):
   with self.assertRaises(ValueError):runtime.binding(Path('/tmp/fixture/report.json'),'a'*64)
  proof=self.proof();proof['runtime_envelope']['injection']['sha256']='15c1'+'0'*60
  with patch.object(fixture,'finalized_binding',return_value=proof):
   with self.assertRaises(ValueError):runtime.binding(Path('/tmp/fixture/report.json'),'a'*64)
 def test_narrow_actor_delta(self):
  here=Path(actor.__file__).parent
  old=(here/'run_source43_warm_pti_v2.py').read_text();new=Path(actor.__file__).read_text()
  old=old.replace('source43_sampled_pti_runtime_v1 as fresh_tool','source43_paused_pti_runtime_v1 as fresh_tool').replace('import pti_temporal_fixture_binding_v2 as fixture\n','import source43_profile_normal_history_adapter_v1 as historical\n').replace('source43-warm-pti-source-plan-v2.json','source43-warm-pti-source-plan-v3.json').replace('fixture.binding();base.closure()','base.closure()').replace('from validate_source43_warm_pti_v2 import','from validate_source43_warm_pti_v3 import').replace('import pti_sampled_census_exec_v1 as census','import pti_paused_census_exec_v1 as census').replace('Fresh43 normal model prerequisite required','Named completed normal43 history required; current arm gates remain fresh')
  before=ast.parse(old);after=ast.parse(new)
  for tree in(before,after):
   index=next(i for i,node in enumerate(tree.body)if isinstance(node,ast.FunctionDef)and node.name=='prerequisite');tree.body.pop(index)
  self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
if __name__=='__main__':unittest.main()
