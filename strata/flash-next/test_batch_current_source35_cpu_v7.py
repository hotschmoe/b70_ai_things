"""V7 ARM/budget/native/API/serial integration controls; no runtime/model execution."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import batch_numerical_execution_v7 as q
import qualify_batch_numerical_v7 as parent
import batch_numerical_proofs_v7 as proof
import audit_batch_fidelity_coverage_v6 as migration
HERE=Path(__file__).resolve().parent

def ast_function(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)

class V7Tests(unittest.TestCase):
 def test_pertransport_native32_API64_serial1_declared(self):
  for n in (2,4,6):
   for kind,want in [('native',[32]*n),('api',[64]*n),('serial',[1]*n)]:
    budget=q.output_budget_contract(kind,n,[32]*n,[64]*n,n if kind=='serial' else None);self.assertEqual(budget['actual_submission_budget'],want);self.assertFalse(budget['natural_completion_qualified']);self.assertFalse(budget['native_64_claim']);self.assertEqual(budget['serial_first_head_max_new'],1)
 def test_plan64_cannot_masquerade_as_native32(self):
  for n in (2,4,6):
   for wrong in ([64]*n,[16]*n,[32]*(n-1),[True]*n):
    with self.assertRaises(ValueError):q.output_budget_contract('native',n,wrong,[64]*n)
   with self.assertRaises(ValueError):q.output_budget_contract('api',n,[32]*n,[32]*n)
 def test_serial_budget_matches_actual_group4_not_logical_slots2(self):
  budget=q.output_budget_contract('serial',2,[32,32],[64,64],4);self.assertEqual(budget['actual_submission_budget'],[1]*4);self.assertEqual(budget['actual_serial_group_job_count'],4)
  with self.assertRaises(ValueError):q.output_budget_contract('serial',2,[32,32],[64,64],0)
 def test_parent_CPU_admission_precedes_lease_and_health(self):
  text=(HERE/'qualify_batch_numerical_v7.py').read_text();preflight=text.index('preflight_plan=read(a.plan);prepared_chain_binding(preflight_plan)');self.assertLess(preflight,text.index('if not a.leased:'));self.assertLess(preflight,text.index("pre=health('pre')"))
 def test_optimized_python_refused_before_registry_baseline_or_GPU(self):
  with patch.dict(q.os.environ,{'PYTHONOPTIMIZE':'1'}),patch.object(q,'genuine_baseline') as baseline:
   with self.assertRaisesRegex(ValueError,'assertions enabled'):q.manifest_binding({})
   baseline.assert_not_called()
 def test_newspec_corpus_unchanged_budgets_separate(self):
  for n in (2,4,6):
   old=json.loads((HERE/f'batch-numerical-case{n}-source35-v1.json').read_text());new=json.loads((HERE/f'batch-numerical-case{n}-source35-v2.json').read_text());self.assertEqual(new['schema'],2);self.assertEqual(new['harness_generation'],7);self.assertEqual(new['native_diagnostic_max_new_by_request'],[32]*n);self.assertEqual(new['api_max_new_by_request'],old['max_new_by_request'])
   for name in ('tokens','messages','api_token_ids','actual_counter_policy','tokenizer_sha256','source_sha256','cancel_index'):self.assertEqual(new[name],old[name])
 def test_oldplan_rejected_before_genuine_baseline_or_payload(self):
  with patch.object(q,'genuine_baseline') as actual:
   with self.assertRaisesRegex(ValueError,'FrozenV6'):q.manifest_binding({'env':{},'schema':3,'harness_generation':6})
   actual.assert_not_called()
 def test_native64_and_partialclosure_rejected_before_baseline(self):
  plan={'env':{},'schema':4,'harness_generation':7,'kind':'native','slots':2,'native_diagnostic_max_new_by_request':[64,64],'api_max_new_by_request':[64,64]}
  with patch.object(q,'genuine_baseline') as actual:
   with self.assertRaises(ValueError):q.manifest_binding(plan)
   plan['native_diagnostic_max_new_by_request']=[32,32];plan.update(output_budget_contract=q.output_budget_contract('native',2,[32,32],[64,64]),max_new_by_request=[32,32],max_new=32,dependency_sha256={})
   with self.assertRaisesRegex(ValueError,'Complete immutable V7'):q.manifest_binding(plan)
   actual.assert_not_called()
 def test_actual_parent_invalid_preflight_has_no_lease_child_or_health(self):
  import io
  from contextlib import redirect_stdout
  from unittest.mock import Mock
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);plan=root/'old-plan.json';plan.write_text(json.dumps({'env':{},'schema':3,'harness_generation':6,'lane':'source35','driver_sha256':q.sha(Path(q.__file__))}));stdout=io.StringIO()
   with patch.object(parent.sys,'argv',['qualifier','--plan',str(plan),'--output',str(root/'unused')]),patch.object(parent.os,'execv') as lease,patch.object(parent.subprocess,'Popen') as process,redirect_stdout(stdout):code=parent.main()
   self.assertEqual(code,1);lease.assert_not_called();process.assert_not_called();self.assertFalse((root/'unused').exists());self.assertIn('CPU preflight before GPU',stdout.getvalue());self.assertIn('ValueError',stdout.getvalue())
 def test_source35_sdk_admission_helpers_unchanged(self):
  for name in ('lane_contract','providers','engine_binding','genuine_baseline','source_observers_off','exact_producer_counters','native_row_event','owner_proofs','terminal_api_join','off_on_histories'):
   self.assertEqual(ast_function(HERE/'batch_numerical_proofs_v6.py',name),ast_function(HERE/'batch_numerical_proofs_v7.py',name))
 def test_health_identity_teardown_parent_helpers_unchanged(self):
  for name in ('stat_signature','full_buffered_identity','finalizable','source_plan_binding'):
   self.assertEqual(ast_function(HERE/'qualify_batch_numerical_v6.py',name),ast_function(HERE/'qualify_batch_numerical_v7.py',name))
  text=(HERE/'qualify_batch_numerical_v7.py').read_text()
  for marker in ("pass_fds=(8,9)","ctrl.c1.leased([0,1])","source_watch('during')","health('post')","full_buffered_identity(","failure_diagnostic"):
   self.assertIn(marker,text)
 def test_newdriver_is_previous_frozen_V7_byte_exact(self):
  self.assertEqual(q.sha(HERE/'run_batch_numerical_pilot_v7.py'),'007cfc79109c441eb45b5ed5fe115d8c265f5d563889768df71db3c0de59c9ab')
 def test_newmigration_preserves_absolute_lines_and_strict_solo_gates(self):
  old=ast.parse((HERE/'audit_batch_fidelity_coverage_v4.py').read_text());new=ast.parse((HERE/'audit_batch_fidelity_coverage_v6.py').read_text());self.assertEqual(ast_function(HERE/'audit_batch_fidelity_coverage_v4.py','parsed'),ast_function(HERE/'audit_batch_fidelity_coverage_v6.py','parsed'))
  text=(HERE/'audit_batch_fidelity_coverage_v6.py').read_text()
  for marker in ("scoped_trace(trace,len(expected))","number+boundary['ARM_line']","warm_prefix+'\\n'+'\\n'.join(normal)","close_at<number","actual==required","Migration needs"):
   if marker=='Migration needs':continue
   self.assertIn(marker,text)
 def test_closure_native_API_serial_and_finalaudit_allV7(self):
  for name in ('batch_numerical_execution_v7.py','qualify_batch_numerical_v7.py','run_batch_numerical_pilot_v7.py','run_batch_api_controls_v7.py','run_batch_serial_controls_v7.py','audit_batch_numerical_suite_v7.py','batch_numerical_proofs_v7.py','audit_batch_fidelity_coverage_v5.py','audit_batch_fidelity_coverage_v6.py'):
   self.assertIn(name,q.DEPENDENCIES);self.assertTrue((HERE/name).is_file())
  self.assertEqual(proof.initial_coverage.__module__,'audit_batch_fidelity_coverage_v5');self.assertEqual(proof.migration_coverage.__module__,'audit_batch_fidelity_coverage_v6')
if __name__=='__main__':unittest.main()
