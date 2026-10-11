import copy,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import audit_source43_t1_conv_commit_v2 as q
import test_source43_t1_conv_commit_cpu_v1 as old
class Controls(old.Controls):
 def setUp(self):self.p=patch.object(old,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class CompactControls(old.Controls):
 # The original eight run once above; these are compact-schema additions.
 test_exact_channel_history_commit_including_negative_zero=None;test_nonzero_before_refused=None;test_wrong_history_column=None;test_transposed_layout_not_accepted=None;test_swapped_channel_corruption=None;test_consumed_raw_mutation_and_alias=None;test_actual_shaped_four_request_saved_descriptor_audit=None;test_wrong_corrected_reader_and_artifact_hash=None
 def setUp(self):self.p=patch.object(old,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
 def compact(self,root):
  actor,h,d=self.fixture(root);c={'capture_root':d['capture_root'],'source_generation':43,'materialized_mode':1,'engine_pid':d['engine_pid'],'plan_sha256':d['plan_sha256'],'binding_sha256':d['binding_sha256'],'public_report_sha256':h,'public_source_plan_sha256':q.ACTOR_SOURCE,'full_descriptor_sha256':'c'*64,'requests':[{'ordinal':r['ordinal'],'input_ids':r['raw']['ids'],'output_ids':[1],'raw_request_sha256':'d'*64,'metadata':r['layer0']['metadata'],'metadata_sha256':r['layer0']['metadata_sha256'],'raw33':r['layer0']['observed']}for r in d['requests']]};r={'passed':True,'errors':[],'source_plan_sha256':q.CLI2_SOURCE,'actor_root':str(root),'expected_actor_report_sha256':h,'actual_reference_computation_attempted':True,'current43_capture_before':copy.deepcopy(c),'current43_capture_after':c};return actor,h,r
 def test_actual_compact_schema_same_exact_property_no_fake_descriptor(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,h,r=self.compact(root);v=q.audit_compact(root,{'sha256':h,'value':actor},r);self.assertTrue(v['all_four_exact']);self.assertFalse(v['full_public_descriptor_reconstructed']);self.assertFalse(v['public_descriptor_reexecuted']);self.assertEqual(len(v['consumed_evidence']),16)
 def test_compact_beforeafter_mutation_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,h,r=self.compact(root);r['current43_capture_after']['requests'][2]['input_ids']=[8678]
   with self.assertRaises(ValueError):q.audit_compact(root,{'sha256':h,'value':actor},r)
 def test_wrong_original_CLI_source_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,h,r=self.compact(root);r['source_plan_sha256']='old'
   with self.assertRaises(ValueError):q.audit_compact(root,{'sha256':h,'value':actor},r)
 def test_original_actor_artifact_mutation_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,h,r=self.compact(root);actor['artifacts'][next(iter(actor['artifacts']))]='changed'
   with self.assertRaises(ValueError):q.audit_compact(root,{'sha256':h,'value':actor},r)
class OutputControls(unittest.TestCase):
 def test_external_alias_into_original_refuses_before_any_write(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor=root/'actor';actor.mkdir();alias=root/'alias';alias.symlink_to(actor,target_is_directory=True);out=alias/'new-audit'
   with patch.object(q,'source_binding',side_effect=AssertionError('Source/read boundary entered')):
    with self.assertRaises(ValueError):q.run(actor,'a'*64,root/'missing','b'*64,out,compact=True)
   self.assertFalse((actor/'new-audit').exists());self.assertEqual(list(actor.iterdir()),[])
 def test_alias_input_root_refused_before_mkdir(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor=root/'actor';actor.mkdir();alias=root/'alias';alias.symlink_to(actor,target_is_directory=True);out=root/'new-audit'
   with self.assertRaises(ValueError):q.run(alias,'a'*64,root/'missing','b'*64,out)
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
