"""Fresh37 admission/ownership/error controls; no actual runtime or model input."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import batch_numerical_proofs_v40 as proof
import batch_numerical_execution_v40 as ctrl
import prepare_batch_case37_v4 as case
import batch40_runtime_evidence_v1 as evidence
class Admission(unittest.TestCase):
 def test_lane_exact_newSDK_C137strongparent_pins(self):
  spec=proof.lane_contract('source37');self.assertEqual((spec['source_files'],spec['headers'],spec['patches'],spec['generation'],spec['parent_generation']),(64,28,37,137,1374));c,p=proof.providers();self.assertEqual(proof.sha(c.__file__),spec['controller_sha256']);self.assertEqual(proof.sha(p.__file__),spec['parent_sha256'])
  for old in ('source35','source33','source31','source29'):
   with self.assertRaises(ValueError):proof.lane_contract(old)
 def test_oldSDK_fails_before_any_sourcebytes_or_ELF(self):
  with patch.object(proof,'read',side_effect=[proof.read(proof.ROOT/proof.PLAN),{'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'plan_sha256':'82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7','image':ctrl.c1.BASE_IMAGE}]):
   with self.assertRaises(ValueError):proof.engine_binding(Path('/CPU_OLD_SOURCE35'))
 def test_actual_case_port_preserves_all_original_corps_and_budgets(self):
  for n in (2,4,6):
   old=proof.read(case.HERE/('batch-numerical-case'+str(n)+'-source35-v2.json'));new=case.prepare(n)
   for key in ('tokens','messages','api_token_ids','actual_counter_policy','tokenizer_sha256','cancel_index','native_diagnostic_max_new_by_request','api_max_new_by_request','max_new_by_request'):self.assertEqual(new[key],old[key],key)
   self.assertEqual(new['source_sha256'],proof.read(proof.ROOT/proof.PLAN)['expected_patched_source_sha256']);self.assertEqual(len(new['source_sha256']),64);self.assertFalse(new['actual_current_source37_runtime_qualified'])
 def test_actual_new_case_gate_then_old_or_subset_rejected(self):
  spec=case.prepare(2);engine=Path('/CPU_SOURCE37');plan=proof.read(proof.ROOT/proof.PLAN)
  def digest(path):return proof.PLAN_SHA if Path(path)==proof.ROOT/proof.PLAN else plan['expected_patched_source_sha256'][str(Path(path).relative_to(engine/'source'))]
  with patch.object(proof,'sha',side_effect=digest):
   self.assertEqual(proof.case_source_gate(spec,engine)['source_count'],64)
   old=copy.deepcopy(spec);old['harness_generation']=7
   with self.assertRaises(ValueError):proof.case_source_gate(old,engine)
   old=copy.deepcopy(spec);old['source_sha256'].pop(next(iter(old['source_sha256'])))
   with self.assertRaises(ValueError):proof.case_source_gate(old,engine)
 def test_freshsameSDK_oneandpair_required(self):
  one={'cards':[0],'engine_receipt_sha256':'CPU_NEW'};pair={'cards':[0,1],'engine_receipt_sha256':'CPU_NEW'}
  with patch.object(proof,'genuine_baseline',side_effect=[(one,{'CPU_ONE':True}),(pair,{'CPU_PAIR':True})]):self.assertEqual(proof.topology_baselines('/CPU_ONE','/CPU_PAIR')['engine_receipt_sha256'],'CPU_NEW')
  pair['engine_receipt_sha256']='CPU_OLD'
  with patch.object(proof,'genuine_baseline',side_effect=[(one,{}),(pair,{})]):
   with self.assertRaises(ValueError):proof.topology_baselines('/CPU_ONE','/CPU_PAIR')
 def test_H36_OFF_and_EAGER_presence_failclosed(self):
  for flag in ('STRATA_CRITICAL_PATH_TRACE','STRATA_PLE_INPUT33','STRATA_PREFIX30'):
   with self.assertRaises(ValueError):proof.source_observers_off({flag:'1'})
  with self.assertRaises(ValueError):proof.source_observers_off({'STRATA_VERIFY_EAGER':'0'})
 def test_oldharness_rejected_before_registry_or_baseline(self):
  with patch.object(ctrl,'genuine_baseline') as baseline,patch.object(ctrl,'sha') as digest:
   with self.assertRaises(ValueError):ctrl.manifest_binding({'env':{},'schema':4,'harness_generation':7})
   baseline.assert_not_called();digest.assert_not_called()
 def test_frozen_batch37_harness_rejected_before_registry_or_baseline(self):
  with patch.object(ctrl,'genuine_baseline') as baseline,patch.object(ctrl,'sha') as digest:
   with self.assertRaises(ValueError):ctrl.manifest_binding({'env':{},'schema':4,'harness_generation':37})
   baseline.assert_not_called();digest.assert_not_called()
 def test_corrected_registry_fragment_has_no_duplicate_models_key(self):
  import yaml
  root=proof.ROOT/'evals/configs/models.yaml';current=yaml.safe_load(root.read_text());fragment=yaml.safe_load((ctrl.HERE/'batch-source37-model-registry-proposal-v2.yaml').read_text())
  self.assertIsInstance(fragment,list);self.assertEqual(len(fragment),12)
  ids=[m['served_model_id'] for m in current['models']]
  self.assertTrue(all(ids.count(m['served_model_id'])==1 for m in fragment))
 def test_native32_API64_serial1_contract_remains_actual(self):
  for n in (2,4,6):
   self.assertEqual(ctrl.output_budget_contract('native',n,[32]*n,[64]*n)['actual_submission_budget'],[32]*n)
   self.assertEqual(ctrl.output_budget_contract('serial',n,[32]*n,[64]*n,4)['actual_submission_budget'],[1]*4)
   with self.assertRaises(ValueError):ctrl.output_budget_contract('native',n,[64]*n,[64]*n)
 def test_child_paths_proofimports_and_runtime_gates_sourcebound(self):
  source=(ctrl.HERE/'run_batch_numerical_pilot_v40.py').read_text();self.assertIn('from batch_numerical_proofs_v40 import',source);self.assertNotIn('batch_numerical_proofs_v6',source)
  source=(ctrl.HERE/'qualify_batch_numerical_v40.py').read_text()
  for marker in ("row['error'] is None and not FAULT", "parent.setdefault('kernel_journal_receipts',{})[stage]=row", "parent['leaf_log_sha256']", "parent['child_started_epoch']=time.time()", 'pass_fds=(8,9)', "health('post')", 'full_buffered_identity('):self.assertIn(marker,source)
  self.assertEqual(len(ctrl.DEPENDENCIES),len(set(ctrl.DEPENDENCIES)))
 def test_model_and_health_teardown_helpers_exact_V7_ast(self):
  def function(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
  for name in ('stat_signature','full_buffered_identity','finalizable','source_plan_binding'):self.assertEqual(function(ctrl.HERE/'qualify_batch_numerical_v7.py',name),function(ctrl.HERE/'qualify_batch_numerical_v40.py',name))
 def test_progression_before4or6_needs_actual_private_reader_and_full49(self):
  from audit_batch_numerical_suite_v40 import paired_two_control
  with patch('audit_batch_numerical_suite_v40.sha',return_value='CPU_WRONG'):
   with self.assertRaises(ValueError):paired_two_control({'reader_sha256':'a'*64},'CPU_NEW')
  s=(ctrl.HERE/'audit_batch_numerical_suite_v40.py').read_text();self.assertIn("raw=compare_all(batch,serial);require(raw['passed']",s);self.assertIn('private=reader.finalized_binding(off,on)',s)

class Journal(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.parent={'started_epoch':10.,'finished_epoch':40.,'pre_health_finished_epoch':11.,'post_health_finished_epoch':25.,'child_started_epoch':15.,'child_terminal_epoch':20.,'kernel_journal_receipts':{}}
  for stage,start,finish in [('pre',12.,13.),('post',26.,27.)]:
   path=self.root/(stage+'-kernel-journal.log');path.write_text('CPU_SYNTHETIC_COMPLETE\n');cmd=self.root/(stage+'-kernel-journal.command.json');argv=['journalctl','-k','--since','@10','--no-pager'];proof.write(cmd,argv);row={'path':str(path),'sha256':proof.sha(path),'stdout_sha256':proof.sha(path),'command':argv,'command_file_sha256':proof.sha(cmd),'return_code':0,'error':None,'started_epoch':start,'finished_epoch':finish};self.parent['kernel_journal_receipts'][stage]=row;self.save(stage)
 def save(self,stage):proof.write(self.root/(stage+'-kernel-receipt.json'),self.parent['kernel_journal_receipts'][stage])
 def test_complete_original_pre_post(self):
  for stage in ('pre','post'):self.assertEqual(evidence.journal_binding(self.root,self.parent,stage),self.parent['kernel_journal_receipts'][stage])
 def test_rczero_error_and_changedlog_refused(self):
  self.parent['kernel_journal_receipts']['post']['error']='CPU_TIMEOUT';self.save('post')
  with self.assertRaises(ValueError):evidence.journal_binding(self.root,self.parent,'post')
  (self.root/'pre-kernel-journal.log').write_text('CPU_CHANGED\n')
  with self.assertRaises(ValueError):evidence.journal_binding(self.root,self.parent,'pre')
 def test_chronology_nonfinite_and_stale_prelaunch_refused(self):
  self.parent['child_started_epoch']=12.
  with self.assertRaises(ValueError):evidence.journal_binding(self.root,self.parent,'pre')
  self.parent['finished_epoch']=float('inf')
  with self.assertRaises(ValueError):evidence.journal_binding(self.root,self.parent,'post')
 def test_wrong_actual_argv_even_updated_command_sha(self):
  row=self.parent['kernel_journal_receipts']['pre'];row['command'][-1]='CPU_CHANGED';cmd=self.root/'pre-kernel-journal.command.json';proof.write(cmd,row['command']);row['command_file_sha256']=proof.sha(cmd);self.save('pre')
  with self.assertRaises(ValueError):evidence.journal_binding(self.root,self.parent,'pre')
 def test_childsnapshot_interpreter_and_original_log_binding(self):
  childroot=self.root/'child';childroot.mkdir();log=childroot/'engine.combined.log';log.write_text('CPU_ONLY_ACTUAL_LOG\n');interpreter=self.root/'CPU_INTERPRETER';interpreter.write_bytes(b'CPU_ONLY_INTERPRETER_NOT_A_RUNTIME_POSITIVE');planfile=self.root/'plan.json';planfile.write_text('{}')
  p=self.parent;p.update(parent_generation=40,child_pid=123,leaf_logs_epoch=21.,child_interpreter={'path':str(interpreter.resolve()),'sha256':proof.sha(interpreter)},plan=str(planfile),leaf_log_sha256={'engine.combined.log':proof.sha(log)})
  for target,source in [('wrapper.py','qualify_batch_numerical_v40.py'),('controller.py','batch_numerical_execution_v40.py')]: (self.root/target).write_bytes((ctrl.HERE/source).read_bytes())
  cmd=[str(interpreter.resolve()),str(ctrl.HERE/'batch_numerical_execution_v40.py'),'run','--plan',str(planfile),'--pre-health',str(self.root/'pre-health.json'),'--output',str(childroot)];proof.write(self.root/'child.command.json',cmd);p['child_command_sha256']=proof.sha(self.root/'child.command.json');identity=self.root/'post-model-identity.json';proof.write(identity,{'passed':True,'rows':[{}]*4,'started':28.,'finished':29.});p['post_model_identity']={'path':str(identity),'sha256':proof.sha(identity)};child={'finished_epoch':20.,'artifact_bindings':{'engine.combined.log':{'sha256':proof.sha(log)}}}
  observed=evidence.child_binding(self.root,p,{'kind':'native'},child);self.assertEqual(observed['child_pid'],123);self.assertEqual(observed['child_interpreter'],p['child_interpreter']);self.assertEqual(set(observed['child_interpreter']),{'path','sha256'});self.assertNotEqual(observed['child_interpreter'],proof.read(identity))
  proof.write(identity,{'passed':True,'rows':[{}]*4,'started':26.5,'finished':29.});p['post_model_identity']['sha256']=proof.sha(identity)
  with self.assertRaisesRegex(ValueError,'observedpostjournal'):evidence.child_binding(self.root,p,{'kind':'native'},child)
  proof.write(identity,{'passed':True,'rows':[{}]*4,'started':28.,'finished':29.});p['post_model_identity']['sha256']=proof.sha(identity)
  child['finished_epoch']=float('nan')
  with self.assertRaisesRegex(ValueError,'child finish'):evidence.child_binding(self.root,p,{'kind':'native'},child)
  child['finished_epoch']=20.
  log.write_text('CPU_CHANGED\n')
  with self.assertRaises(ValueError):evidence.child_binding(self.root,p,{'kind':'native'},child)
if __name__=='__main__':unittest.main()
