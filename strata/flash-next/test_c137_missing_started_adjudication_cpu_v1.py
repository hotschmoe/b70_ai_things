"""Saved failure metadata and negative views only; no model/device/validator run."""
import copy,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import adjudicate_c137_missing_started_v3_v1 as a
import c137_baseline_admission_v4 as consumer
class Controls(unittest.TestCase):
 def setUp(self):
  self.parent=a.c.read(a.RUN/'parent-qualification.json');self.before=a.c.read(a.RUN/'parent-before-proof.json');self.proof=a.c.read(a.RUN/'c1-source-identity-proof-v137-v3.json')
 def test_actual_saved_metadata_onlymissingfield_view(self):
  p,b=a.metadata_views(self.parent,self.before,self.proof);self.assertFalse(p['passed']);self.assertEqual(p['post_error'],"'started_epoch'");self.assertEqual(set(p)-set(self.parent),{'started_epoch'});self.assertEqual(set(b)-set(self.before),{'started_epoch'});self.assertEqual({k:v for k,v in p.items() if k!='started_epoch'},self.parent);self.assertEqual({k:v for k,v in b.items() if k!='started_epoch'},self.before)
 def test_unknown_error_or_other_errorfield_cannotborrowexception(self):
  for kind in ('error','extra'):
   p=copy.deepcopy(self.parent)
   if kind=='error':p['post_error']='CPU_UNKNOWN_ERROR'
   else:p['cleanup_error']='CPU_OTHER_FAILURE'
   with self.subTest(kind=kind),self.assertRaises(ValueError):a.metadata_views(p,self.before,self.proof)
 def test_changed_original_parent_or_already_present_start_refused(self):
  for kind in ('changed','present'):
   p=copy.deepcopy(self.parent)
   if kind=='changed':p['launch_started_epoch']+=1
   else:p['started_epoch']=self.proof['started_epoch']
   with self.subTest(kind=kind),self.assertRaises(ValueError):a.metadata_views(p,self.before,self.proof)
 def test_nonfinite_saved_start_or_missing_terminal_refused(self):
  proof=copy.deepcopy(self.proof);proof['started_epoch']=float('nan')
  with self.assertRaises(ValueError):a.metadata_views(self.parent,self.before,proof)
  p=copy.deepcopy(self.parent);b=copy.deepcopy(self.before);p['owned_terminal']=b['owned_terminal']=False
  with self.assertRaises(ValueError):a.metadata_views(p,b,self.proof)
 def test_genuine_original_pass_cannotbe_reinterpreted_as_thisfailure(self):
  p=copy.deepcopy(self.parent);p['passed']=True
  with self.assertRaises(ValueError):a.metadata_views(p,self.before,self.proof)
 def test_foreign_run_refused_before_current_model_or_source_validation(self):
  with patch.object(a.c,'validate_prepared') as prepared:
   with self.assertRaises(ValueError):a.finalized_binding('/CPU_FOREIGN')
   prepared.assert_not_called()
 def test_current_original_artifactpins_and_frozen_source_match(self):
  for name,digest in a.PINS.items():self.assertEqual(a.c.sha(a.RUN/name),digest)
  self.assertEqual(a.source_binding()['old_parent_sha256'],a.OLD_SHA)
 def test_tree_unchanged_binding_detects_changed_metadata(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);f=root/'CPU.log';f.write_text('CPU_SAVED\n');old=a.tree_state(root);f.write_text('CPU_CHANGED\n');self.assertNotEqual(old,a.tree_state(root))
 def test_explicit_adjudication_consumer_nevertrusts_passed_flag_only(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'CPU-adjudication.json';saved={'adjudication_kind':'C137_actual90362_missing_started_v3_v1','original_run_root':str(a.RUN),'passed':True};a.c.write(path,saved)
   with patch.object(a,'finalized_binding',return_value=dict(saved,passed=False)):
    with self.assertRaises(ValueError):consumer.admit_adjudication(path)
 def test_saved_output_only_controller_trace_terminal_binding(self):
  raw=a.c.read(a.RUN/'qualification-controller-v137-v3.json');prepared=a.c.read(a.RUN/'prepared.json');self.assertEqual(len(a.raw_controller_binding(a.RUN,raw,prepared)['cases']),6)
 def test_saved_owned_terminal_error_cannotbe_hidden_by_rawPASS(self):
  raw=a.c.read(a.RUN/'qualification-controller-v137-v3.json');prepared=a.c.read(a.RUN/'prepared.json');stop=a.c.read(a.RUN/'stop.json');stop['terminal']['Error']='CPU_OTHER_FAILURE';read=a.c.read
  with patch.object(a.c,'read',side_effect=lambda p:stop if Path(p)==a.RUN/'stop.json' else read(p)):
   with self.assertRaises(ValueError):a.raw_controller_binding(a.RUN,raw,prepared)
if __name__=='__main__':unittest.main()
