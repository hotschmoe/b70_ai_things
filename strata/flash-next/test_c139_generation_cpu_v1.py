"""Synthetic source39 marker/old-proof rejection controls; no GPU/model IO."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c1_serve_controller_combined_v139 as c
import qualify_c1_serving_combined_v139_v1 as parent
import c139_baseline_admission_v1 as admission
import admit_source_upload39_v1 as upload
import prepare_full_cache_memory39_v1 as source
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='C139-source-CPU-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);_,rows=source.reconstruct()
  for name,text in rows.items():
   p=self.root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
  (self.root/'build').mkdir();self.binary=self.root/'build/strata';self.binary.write_bytes(b'CPU_SYNTHETIC_MARKERS STRATA_FULL_CACHE_OBSERVER38 STRATA_FULL_CACHE_CAPTURE_RIDS38 FC38 FC39 observed_owned_snapshot_peak_bytes')
 def test_actual_reconstructed39_source_presence_synthetic_binary_markers(self):
  r=c.compiled_source_markers(self.root);self.assertTrue(r['default_off_full_cache_observer38']);self.assertTrue(r['default_off_logical_cache_memory39']);self.assertTrue(r['earlier_stage_batch_observer_stamp37']);self.assertFalse(r['actual_full_cache38_39_observer_qualified'])
 def test_missing39_binary_marker_or_header_refused(self):
  self.binary.write_bytes(b'CPU_SYNTHETIC FC38')
  with self.assertRaises(ValueError):c.compiled_source_markers(self.root)
 def test_earlier_stamp_wrong_handoff_rejected(self):
  p=self.root/'source/sycl/src/core/verify.cpp';s=p.read_text();hook='if(batch_observe_ && batch_snapshot_)batch_snapshot_->stamp(*cs);';p.write_text(s.replace(hook,'',1))
  with self.assertRaises(ValueError):c.compiled_source_markers(self.root)
 def test_normalOFF_and_selector_absence_eager_presence_reject(self):
  args=['--batch','0'];c.baseline_profile_gate(args,{})
  for env in ({'STRATA_FULL_CACHE_OBSERVER38':'1'},{'STRATA_FULL_CACHE_CAPTURE_RIDS38':''},{'STRATA_VERIFY_EAGER':'0'}):
   with self.assertRaises(ValueError):c.baseline_profile_gate(args,env)
 def test_new66_30_39_8_6_recipe_and_old_plan_digest_differ(self):
  p=c.read(c.COMBINED_PLAN);self.assertEqual((len(p['expected_patched_source_sha256']),len(p['added_header_payloads']),len(p['patches']),len(p['build_targets']),len(p['runtime_python_sources'])),(66,30,39,8,6));self.assertNotEqual(c.COMBINED_PLAN_SHA,'2e940d51c61abe5366526b926f89dcf9f42bf778feee5d13c9b72aefc2785ace')
 def test_upload37_plan_cannot_enter139_before_runtime_access(self):
  with patch.object(c,'read',return_value={'plan_sha256':'4a494a0e38f3bac5e29d4756d8e361be8ce8dd29ce9b85726addc9ba107c6053'}):
   with self.assertRaises(ValueError):c.upload_gate(Path('/CPU_RECEIPT'),Path('/CPU_ORACLE'),Path('/CPU_ENGINE'),Path('/CPU_PACK'))
 def test_old_source37_baseline_or_adjudication_rejected_before_payload(self):
  for row in ({'c1_parent_generation':1374},{'passed':True,'adjudication_kind':'C137_actual90362_missing_started_v3_v1'}):
   with patch.object(c,'read',return_value=row),patch.object(c,'validate_prepared') as forbidden:
    with self.assertRaises(ValueError):admission.finalized_binding(Path('/CPU_BASELINE'))
    forbidden.assert_not_called()
 def test_actual_newupload_plan_baseline_OFF_and_fresh_engine_association(self):
  p=c.read(upload.PLAN);self.assertEqual(p['source_generation'],39);self.assertEqual(p['engine_source_plan']['sha256'],c.COMBINED_PLAN_SHA);self.assertEqual(p['runtime_environment']['STRATA_FULL_CACHE_OBSERVER38'],'0');self.assertFalse(p['old_source37_oracle_or_upload_receipt_transfer_allowed'])
 def test_parent_real_initial_result_started_and1391_generation(self):
  import ast
  tree=ast.parse(Path(parent.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='parent_generation' for k in n.keywords)];self.assertEqual(len(calls),1);keys={k.arg for k in calls[0].keywords};self.assertIn('started_epoch',keys);self.assertEqual(next(k.value.value for k in calls[0].keywords if k.arg=='parent_generation'),1391)
if __name__=='__main__':unittest.main()
