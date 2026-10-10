"""Authenticated real declaration mutations; no actual model/pack reads."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import batch50_case_binding_v1 as case
import batch_numerical_execution_v50 as ctrl
class Controls(unittest.TestCase):
 def plan(self,n=4,kind='native'):
  name,digest=case.SPECS[n];path=case.HERE/name;spec=case.authenticated(path,n);plan={k:copy.deepcopy(spec[k]) for k in case.FIELDS};budget=ctrl.output_budget_contract(kind,n,spec['native_diagnostic_max_new_by_request'],spec['api_max_new_by_request'],2 if kind=='serial' else None)
  plan.update(kind=kind,spec_path=str(path.resolve()),spec_sha256=digest,spec_tokenizer_sha256=spec['tokenizer_sha256'],output_budget_contract=budget,max_new_by_request=budget['actual_submission_budget'],max_new=32)
  if kind=='serial':plan['serial_group_job_count']=2
  return plan
 def test_genuine_frozen4_6_native_serial_case_fields(self):
  for n in (4,6):
   for kind in ('native','serial'):self.assertEqual(case.binding(self.plan(n,kind),ctrl.output_budget_contract)['slots'],n)
 def reject(self,plan):self.assertRaises(ValueError,case.binding,plan,ctrl.output_budget_contract)
 def test_changed_token_retaining_old_spec_sha(self):
  p=self.plan();p['tokens']['target'][0][0]+=1;self.reject(p)
 def test_changed_message(self):
  p=self.plan();p['messages']['target'][0][0]['content']+=' changed';self.reject(p)
 def test_changed_rendered_api_ids(self):
  p=self.plan();p['api_token_ids']['target'][0][0]+=1;self.reject(p)
 def test_changed_cancel(self):
  p=self.plan();p['cancel_index']=(p['cancel_index']+1)%4;self.reject(p)
 def test_changed_counters(self):
  p=self.plan();p['actual_counter_policy'][0]['values']['reused']=1;self.reject(p)
 def test_changed_native_or_api_budget(self):
  for key in ('native_diagnostic_max_new_by_request','api_max_new_by_request'):
   p=self.plan();p[key][0]+=1;self.reject(p)
 def test_changed_actual_submission_budget(self):
  p=self.plan(kind='serial');p['max_new_by_request']=[1,2];self.reject(p)
 def test_changed_port_and_control_and_tokenizer(self):
  for key in ('port','paired_two_control','spec_tokenizer_sha256'):
   p=self.plan();p[key]=0;self.reject(p)
 def test_same_sha_foreign_case_path_refused(self):
  p=self.plan()
  with tempfile.TemporaryDirectory() as d:
   other=Path(d)/'case';other.write_bytes(Path(p['spec_path']).read_bytes());p['spec_path']=str(other);self.reject(p)
 def test_changed_current_spec_bytes_or_declared_sha(self):
  p=self.plan();p['spec_sha256']='0'*64;self.reject(p)
  p=self.plan()
  with patch.object(case,'sha',return_value='0'*64):self.reject(p)
 def test_bool_cancel_does_not_equal_int(self):
  p=self.plan();p['cancel_index']=bool(p['cancel_index']);self.reject(p)
 def test_corpus_mutation_fails_manifest_before_any_baseline(self):
  p=self.plan();p.update(env={},slots=4,binding_type_diagnosis={'path':'CPU_synthetic'});p['tokens']['target'][0][0]+=1
  with patch.object(ctrl,'diagnosis_binding',return_value={'path':'CPU_synthetic'}),patch.object(ctrl,'topology_baselines',side_effect=AssertionError('must not reach costly/device prerequisite')):self.assertRaises(ValueError,ctrl.manifest_binding,p)
if __name__=='__main__':unittest.main()
