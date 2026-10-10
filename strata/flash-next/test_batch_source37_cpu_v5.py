"""Source44 prospective4/6 control and cheap-prelease ordering, no payload/GPU."""
import ast,copy,json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_numerical_execution_v44 as ctrl
import batch_numerical_proofs_v44 as proof
import paired_source37_private_v4_control_v1 as control
import prepare_batch_case37_v5 as case
import batch44_prelease_v1 as prelease
class Controls(unittest.TestCase):
 def test_new4_6_cases_keep_exact_corpus_geometry_budgets(self):
  for n in (4,6):
   new=case.prepare(n);old=proof.read(case.HERE/('batch-numerical-case'+str(n)+'-source35-v2.json'))
   for key in ('tokens','messages','api_token_ids','actual_counter_policy','tokenizer_sha256','cancel_index','native_diagnostic_max_new_by_request','api_max_new_by_request'):self.assertEqual(new[key],old[key],key)
   self.assertEqual(new['harness_generation'],44);self.assertEqual(len(new['source_sha256']),64)
  with self.assertRaises(ValueError):case.prepare(2)
 def test_wrong_or_missing_private4_decl_before_actual_proof(self):
  for declaration in ({},{'schema':3},{'schema':4}):
   with self.assertRaises(ValueError):control.finalized_binding(declaration,'CPU_ENGINE')
 def test_private4_helper_and_actual_closed_report_sourcepins(self):
  self.assertEqual(control.sha(control.HERE/'validate_private_native_offon_source37_v4.py'),control.READER_SHA);self.assertEqual(control.sha(control.HERE/'private-native-offon-source37-source-plan-v4.json'),control.READER_PLAN_SHA)
  self.assertEqual(control.REPORT_SHA,'c05d9416ac7b92a78f9d770283c22dd118017a4ef6281c2740b46eb6eb4bc9f1')
 def fake(self,root,events):
  driver=root/'driver.py';driver.write_text('CPU_driver');dep=root/'helper.py';dep.write_text('CPU_helper')
  digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  def require(ok,message):
   if not ok:raise ValueError(message)
  def manifest(plan):
   events.append('full');require(digest(dep)==plan['dependency_sha256']['helper.py'],'Mutation caught before any device');return {'CPU_synthetic_current_admission':True}
  c=SimpleNamespace(__file__=str(driver),sha=digest,require=require,HERE=root,DEPENDENCIES=['helper.py'],source_observers_off=lambda env:require('STRATA_VERIFY_EAGER' not in env,'EAGER'),c1=SimpleNamespace(leased=lambda cards:events.append('lease')),manifest_binding=manifest)
  plan={'schema':4,'harness_generation':44,'slots':4,'driver_sha256':digest(driver),'dependency_sha256':{'helper.py':digest(dep)},'env':{}}
  return c,plan,dep
 def test_cheap_does_not_call_full_or_lease_or_model(self):
  with tempfile.TemporaryDirectory(prefix='h44-cheap-CPU-') as d:
   events=[];c,p,dep=self.fake(Path(d),events);r=prelease.cheap(p,c);self.assertFalse(r['model_or_pack_payload_read']);self.assertEqual(events,[])
 def test_current_mutation_between_cheap_and_full_rejected_predevice(self):
  with tempfile.TemporaryDirectory(prefix='h44-mutation-CPU-') as d:
   events=[];c,p,dep=self.fake(Path(d),events);prelease.cheap(p,c);dep.write_text('CPU_MUTATED')
   with self.assertRaises(ValueError):prelease.full_leased(p,c)
   self.assertEqual(events,['lease','full'])
 def test_one_leased_full_then_health_order(self):
  with tempfile.TemporaryDirectory(prefix='h44-order-CPU-') as d:
   events=[];c,p,dep=self.fake(Path(d),events);prelease.cheap(p,c);prelease.full_leased(p,c);events.append('CPU_mock_health');self.assertEqual(events,['lease','full','CPU_mock_health'])
 def test_source_parent_has_exactly_one_full_and_no_startup_repeat(self):
  s=(ctrl.HERE/'qualify_batch_numerical_v44.py').read_text();self.assertEqual(s.count('current_chain=full_leased(preflight_plan,ctrl)'),1);self.assertIn("parent['prepared_chain']=current_chain",s);self.assertNotIn("parent['prepared_chain']=prepared_chain_binding(plan)",s);self.assertLess(s.index('current_chain=full_leased'),s.index("pre=health('pre')"));self.assertIn('cheap_prelease(preflight_plan,ctrl)',s)
 def test_new_serial_extract_absentPIN_rawfile_plan_SHA_and_EOF(self):
  s=(ctrl.HERE/'run_batch_serial_controls_v44.py').read_text();self.assertIn('request=AbsentPin.request',s);self.assertIn('meta=extract(raw',s);self.assertNotIn('numerical.base.extract(',s);self.assertIn("plan_sha256=sha(a.output/'plan.snapshot.json')",Path(ctrl.__file__).read_text());self.assertIn('stdout_binding(root,parent)',(ctrl.HERE/'batch44_runtime_evidence_v1.py').read_text());self.assertIn('Actual individual serial producer differs',(ctrl.HERE/'audit_batch_numerical_suite_v44.py').read_text())
 def test_normalgraph_flags_and_oldSDK_count(self):
  self.assertEqual((proof.lane_contract('source37')['source_files'],proof.lane_contract('source37')['headers'],proof.lane_contract('source37')['patches']),(64,28,37))
  for env in ({'STRATA_VERIFY_EAGER':'0'},{'STRATA_CRITICAL_PATH_TRACE':'1'}):
   with self.assertRaises(ValueError):proof.source_observers_off(env)
if __name__=='__main__':unittest.main()
