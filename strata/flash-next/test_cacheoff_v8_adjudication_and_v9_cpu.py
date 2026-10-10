"""Tiny closed-evidence missing-field derivation and future realSHA producer tests."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import adjudicate_batch_serial_cacheoff_v8_v1 as q
import batch_serial_cacheoff_v9 as future

class ClosedTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='closedV8-CPU-');self.root=Path(self.tmp.name);(self.root/'child').mkdir();self.external=self.root/'external-plan.json';self.controller=Path(q.ctrl.__file__);wrapper=Path(q.__file__).with_name('qualify_batch_serial_cacheoff_v8.py');self.plan={'driver_sha256':q.sha(self.controller),'CPU_SYNTHETIC':True}
  for path in (self.external,self.root/'input-plan.snapshot.json',self.root/'child/plan.snapshot.json'):self.write(path,self.plan)
  self.child={'passed':True,'CPU_SYNTHETIC':True};self.write(self.root/'child/report.json',self.child);self.parent={'passed':True,'plan':str(self.external),'plan_sha256':q.sha(self.external),'child_report_sha256':q.sha(self.root/'child/report.json'),'controller_sha256':q.sha(self.controller),'wrapper_sha256':q.sha(wrapper)};self.write(self.root/'parent-qualification.json',self.parent);(self.root/'controller.py').write_bytes(self.controller.read_bytes());(self.root/'wrapper.py').write_bytes(wrapper.read_bytes());self.command=['/usr/bin/python3',str(self.controller.resolve()),'run','--plan',str(self.external),'--pre-health',str(self.root/'pre-health.json'),'--output',str(self.root/'child')];self.write(self.root/'child.command.json',self.command)
 def tearDown(self):self.tmp.cleanup()
 def write(self,path,value):path.write_text(json.dumps(value,indent=2)+'\n')
 def test_missing_field_view_explicit_only_without_original_mutation(self):
  before=q.tree_binding(self.root);view,provenance=q.derive_child_view(self.root,self.parent,self.child);self.assertNotIn('plan_sha256',self.child);self.assertEqual(view['plan_sha256'],q.sha(self.root/'child/plan.snapshot.json'));self.assertFalse(provenance['producer_emitted_missing_field']);self.assertTrue(provenance['original_child_field_absent']);self.assertEqual(before,q.tree_binding(self.root))
 def test_existing_wrong_or_correct_childSHA_not_adjudicated(self):
  for value in ('0'*64,q.sha(self.root/'child/plan.snapshot.json')):
   with self.assertRaises(ValueError):q.derive_child_view(self.root,self.parent,dict(self.child,plan_sha256=value))
 def test_snapshot_external_or_actualCLI_mismatch_refused(self):
  self.command[4]='WRONG_ACTUAL_PLAN';self.write(self.root/'child.command.json',self.command)
  with self.assertRaises(ValueError):q.derive_child_view(self.root,self.parent,self.child)
 def test_later_reader_interpreter_neednot_equal_originalproducer(self):
  with patch.object(q.sys,'executable','/CPU_DIFFERENT_READER_PYTHON'):
   view,provenance=q.derive_child_view(self.root,self.parent,self.child)
  self.assertEqual(provenance['original_producer_interpreter_path'],'/usr/bin/python3');self.assertFalse(provenance['original_interpreter_ELF_hash_recorded_by_V8'])
 def test_controller_or_wrapper_or_originalreport_hash_mismatch_refused(self):
  for filename in ('controller.py','wrapper.py','child/report.json'):
   path=self.root/filename;old=path.read_bytes();path.write_bytes(old+b'X')
   with self.assertRaises(ValueError):q.derive_child_view(self.root,self.parent,self.child)
   path.write_bytes(old)
 def test_tree_symlink_or_stat_content_change_refused_or_detected(self):
  before=q.tree_binding(self.root);self.external.write_bytes(self.external.read_bytes()+b' ');self.assertNotEqual(before,q.tree_binding(self.root));(self.root/'alias').symlink_to(self.external)
  with self.assertRaises(ValueError):q.tree_binding(self.root)
 def test_readonly_adjudicator_injects_only_view_and_restores_reader(self):
  original=q.ctrl.read;seen=[]
  def strict(root):
   child=q.ctrl.read(root/'child/report.json');seen.append(child);return self.parent,child,self.plan,{'matched_vector_pairs':196}
  with patch.object(q,'source_binding',return_value={'CPU':True}),patch.object(q.frozen_reader,'finalized_binding',strict):r=q.adjudicate(self.root)
  self.assertTrue(r['adjudicated_passed']);self.assertEqual(r['actual_matched_vector_pairs'],196);self.assertTrue(r['original_tree_unchanged']);self.assertFalse(r['cache_qualification_granted']);self.assertIs(q.ctrl.read,original);self.assertNotIn('plan_sha256',q.read(self.root/'child/report.json'))
 def test_original_failed_parent_never_repaired(self):
  self.parent['passed']=False;self.write(self.root/'parent-qualification.json',self.parent)
  with patch.object(q,'source_binding',return_value={}):
   with self.assertRaises(ValueError):q.adjudicate(self.root)
 def test_future_controller_emits_real_snapshot_SHA_and_actual_count(self):
  collector=self.root/'collector';(collector/'child').mkdir(parents=True);self.write(collector/'child/serial-jobs.json',{'jobs':[{'rid':i} for i in range(4)]});out=self.root/'new-future';out.mkdir();plan={'batch_parent':str(collector),'group_index':0,'CPU_NEW_SOURCE':True};result={'passed':True,'comparisons':{str(i):{'bitwise_equal':True} for i in range(196)}}
  with patch.object(future.base.proof,'artifact_bindings',return_value={'CPU_TINY':True}):r=future.completed_report(plan,out,result)
  self.assertEqual(r['plan_sha256'],future.sha(out/'plan.snapshot.json'));self.assertEqual(r['actual_serial_job_count'],4);self.assertEqual(r['actual_matched_full49_vector_pairs'],196);self.assertEqual(r['future_controller_generation'],9);self.assertFalse(r['cache_qualification_granted'])
  with patch.object(future.base.proof,'artifact_bindings',return_value={}):
   with self.assertRaises(ValueError):future.completed_report(plan,out,{'passed':True,'comparisons':{'onlyone':{}}})
if __name__=='__main__':unittest.main()
