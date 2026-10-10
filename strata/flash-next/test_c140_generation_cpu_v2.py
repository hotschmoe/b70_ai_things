"""Synthetic source40 marker/old-proof rejection controls; no GPU/model IO."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c1_serve_controller_combined_v140_v2 as c
import qualify_c1_serving_combined_v140_v2 as parent
import c140_baseline_admission_v2 as admission
import admit_source_upload40_v2 as upload
import prepare_native_layer3_qsa_observer_v2 as source
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='C140-source-CPU-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);_,rows=source.reconstruct()
  for name,text in rows.items():
   p=self.root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
  (self.root/'build').mkdir();self.binary=self.root/'build/strata';self.binary.write_bytes(b'CPU_SYNTHETIC_MARKERS STRATA_FULL_CACHE_OBSERVER38 STRATA_FULL_CACHE_CAPTURE_RIDS38 FC38 FC39 observed_owned_snapshot_peak_bytes STRATA_QSA3_TARGET QSA3 frame')
 def test_actual_reconstructed40_source_presence_synthetic_binary_markers(self):
  r=c.compiled_source_markers(self.root);self.assertTrue(r['default_off_full_cache_observer38']);self.assertTrue(r['default_off_logical_cache_memory39']);self.assertTrue(r['earlier_stage_batch_observer_stamp37']);self.assertFalse(r['actual_full_cache38_39_observer_qualified'])
 def test_missing40_binary_marker_or_header_refused(self):
  self.binary.write_bytes(b'CPU_SYNTHETIC FC38')
  with self.assertRaises(ValueError):c.compiled_source_markers(self.root)
 def test_earlier_stamp_wrong_handoff_rejected(self):
  p=self.root/'source/sycl/src/core/verify.cpp';s=p.read_text();hook='if(batch_observe_ && batch_snapshot_)batch_snapshot_->stamp(*cs);';p.write_text(s.replace(hook,'',1))
  with self.assertRaises(ValueError):c.compiled_source_markers(self.root)
 def test_normalOFF_and_selector_absence_eager_presence_reject(self):
  args=['--batch','0'];c.baseline_profile_gate(args,{})
  for env in ({'STRATA_FULL_CACHE_OBSERVER38':'1'},{'STRATA_FULL_CACHE_CAPTURE_RIDS38':''},{'STRATA_VERIFY_EAGER':'0'},{'STRATA_QSA3_TARGET':'1'},{'STRATA_QSA3_TARGET_DIR':''},{'STRATA_QSA3_TARGET_BINDING_SHA256':''}):
   with self.assertRaises(ValueError):c.baseline_profile_gate(args,env)
 def test_new67_31_40_8_6_recipe_and_old_plan_digest_differ(self):
  p=c.read(c.COMBINED_PLAN);self.assertEqual((len(p['expected_patched_source_sha256']),len(p['added_header_payloads']),len(p['patches']),len(p['build_targets']),len(p['runtime_python_sources'])),(67,31,40,8,6));self.assertNotEqual(c.COMBINED_PLAN_SHA,'2e940d51c61abe5366526b926f89dcf9f42bf778feee5d13c9b72aefc2785ace')
 def test_upload37_plan_cannot_enter140_before_runtime_access(self):
  with patch.object(c,'read',return_value={'plan_sha256':'4a494a0e38f3bac5e29d4756d8e361be8ce8dd29ce9b85726addc9ba107c6053'}):
   with self.assertRaises(ValueError):c.upload_gate(Path('/CPU_RECEIPT'),Path('/CPU_ORACLE'),Path('/CPU_ENGINE'),Path('/CPU_PACK'))
 def test_old_source37_baseline_or_adjudication_rejected_before_payload(self):
  for row in ({'c1_parent_generation':1374},{'passed':True,'adjudication_kind':'C137_actual90362_missing_started_v3_v1'}):
   with patch.object(c,'read',return_value=row),patch.object(c,'validate_prepared') as forbidden:
    with self.assertRaises(ValueError):admission.finalized_binding(Path('/CPU_BASELINE'))
    forbidden.assert_not_called()
 def test_actual_newupload_plan_baseline_OFF_and_fresh_engine_association(self):
  p=c.read(upload.PLAN);self.assertEqual(p['source_generation'],40);self.assertEqual(p['engine_source_plan']['sha256'],c.COMBINED_PLAN_SHA);self.assertEqual(p['runtime_environment']['STRATA_FULL_CACHE_OBSERVER38'],'0');self.assertFalse(p['old_source37_oracle_or_upload_receipt_transfer_allowed'])
 def test_parent_real_initial_result_started_and1402_generation(self):
  import ast
  tree=ast.parse(Path(parent.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='parent_generation' for k in n.keywords)];self.assertEqual(len(calls),1);keys={k.arg for k in calls[0].keywords};self.assertIn('started_epoch',keys);self.assertEqual(next(k.value.value for k in calls[0].keywords if k.arg=='parent_generation'),1402)
 def test_parent_closure_runs_and_new_controller_command_is_exact(self):
  parent.source_pins()
  text=Path(parent.__file__).read_text();self.assertIn("ctrl = ROOT / 'strata/flash-next/c1_serve_controller_combined_v140_v2.py'",text)
 def test_registry_fragment_is_only_four_new_exact_profiles(self):
  import yaml
  rows=yaml.safe_load((c.REPO/'strata/flash-next/c1-combined-v140-model-registry-proposal-v1.yaml').read_text());self.assertIsInstance(rows,list);self.assertEqual(len(rows),4)
  self.assertEqual([r['served_model_id'] for r in rows],[p['alias'] for p in c.PROFILES.values()]);self.assertTrue(all(r['primary_client_id']=='hotschmoe-dd' for r in rows))
 def test_old39_build_rejected_before_current_source_reads(self):
  with patch.object(c,'read',side_effect=[c.read(c.COMBINED_PLAN),{'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'image':c.BASE_IMAGE,'plan_sha256':'OLD39'}]):
   with self.assertRaises(ValueError):c.combined_generation_gate(self.root)
 def test_failed_source40V1_init_predicates_rejected(self):
  path=self.root/'source/sycl/src/core/verify.cpp';text=path.read_text();start=text.index('    if(qsa3_target::settings().enabled) {');end=text.index('    if (std::getenv("STRATA_VERIFY_DEBUG")',start);block=text[start:end].replace('!kernels::native_qsa_enabled()','!native_qsa_enabled()');path.write_text(text[:start]+block+text[end:])
  with self.assertRaises(ValueError):c.compiled_source_markers(self.root)
 def test_namespace_only_undo_matches_failedV1_and_pristine_patch(self):
  import prepare_native_layer3_qsa_observer_v1 as old
  _,before=old.reconstruct();_,after=source.reconstruct();name='sycl/src/core/verify.cpp';text=after[name];start=text.index('    if(qsa3_target::settings().enabled) {');end=text.index('    if (std::getenv("STRATA_VERIFY_DEBUG")',start);block=text[start:end]
  for method in ('native_qsa_enabled','native_rope_enabled','native_qsa_indexer_enabled'):block=block.replace('!kernels::'+method+'()','!'+method+'()')
  self.assertEqual(text[:start]+block+text[end:],before[name]);self.assertEqual({k for k in after if after[k]!=before[k]},{name});self.assertEqual(source.patch_bytes(),source.PATCH.read_bytes());self.assertEqual(source.build_plan(),c.read(c.COMBINED_PLAN))
 def test_old1401_parent_rejected_before_SDK_model_access(self):
  with patch.object(c,'read',return_value={'c1_parent_generation':1401}),patch.object(c,'validate_prepared') as forbidden:
   with self.assertRaises(ValueError):admission.finalized_binding(Path('/CPU_BASELINE'))
   forbidden.assert_not_called()
if __name__=='__main__':unittest.main()
