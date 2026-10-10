"""Owned seven-field adapter controls; no runtime/model/old-proof execution."""
import ast,copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import native_rms_device_ops_lifecycle_v8 as life
import native_rms_owned_device_ops_binding_v8 as binding
import qualify_native_rms_rsqrt37_v8 as q
import test_native_rms_device_ops_cpu_v1 as fields
class Tests(unittest.TestCase):
 def test_sevenfield_marker_required_fourfield_rejected(self):
  import test_native_rms_lifecycle_cpu_v5 as original
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'runtime.log';text='\n'.join(original.Tests().markers())+'\n';p.write_text(text.replace('fields=4','fields=7'));self.assertTrue(life.native_trace(p)['actual_unique_frame_free_markers']);p.write_text(text)
   with self.assertRaises(ValueError):life.native_trace(p)
 def test_lifecycle_only_marker_geometry_changes(self):
  h=Path(q.__file__).parent;self.assertEqual((h/'native_rms_device_ops_lifecycle_v8.py').read_bytes(),(h/'native_rms_lifecycle_binding_v5.py').read_bytes().replace(b'fields=4 synthetic_norm=1',b'fields=7 synthetic_norm=1'))
 def test_wrong_prior_root_rejected_before_source_admission_or_lease(self):
  with patch('sys.argv',['qualifier','--fixture','/CPU','--prepared','/CPU','--output','/CPU','--prior-rms-root','/WRONG']),patch.object(q,'source_binding',side_effect=AssertionError('expensive source touched')) as source,patch.object(q.os,'execv') as lease:
   with self.assertRaises(ValueError):q.main()
   source.assert_not_called();lease.assert_not_called()
 def test_prior_actual_report_externalSHA_and_typed_receipt_join(self):
  proof={'root':str(binding.PRIOR_ROOT),'finalized_binding':{'report_sha256':binding.REPORT_SHA,'actual_RMS_observed':True}}
  with patch.object(binding,'header',return_value=binding.PRIOR_ROOT),patch.object(binding.ops,'prior_binding',return_value=copy.deepcopy(proof)),patch.object(binding,'read_unique',side_effect=[proof['finalized_binding'],{'fixture_binding':{'exact':'ORIGINAL'}}]):self.assertEqual(binding.admit_prior(binding.PRIOR_ROOT)['external_binding_sha256'],binding.EXTERNAL_SHA)
  for changed in ('report','external'):
   bad=copy.deepcopy(proof)
   if changed=='report':bad['finalized_binding']['report_sha256']='WRONG'
   with patch.object(binding,'header',return_value=binding.PRIOR_ROOT),patch.object(binding.ops,'prior_binding',return_value=bad),patch.object(binding,'read_unique',return_value={'report_sha256':'WRONG'}):
    with self.assertRaises(ValueError):binding.admit_prior(binding.PRIOR_ROOT)
 def test_fixture_inputs_cannot_be_substituted(self):
  binding.match_fixture({'fixture_binding':{'id':1}},{'id':1})
  for value in (True,1.,2):
   with self.assertRaises(ValueError):binding.match_fixture({'fixture_binding':{'id':1}},{'id':value})
 def test_source_flags_and_both_recipe_source_selections(self):
  p=types.SimpleNamespace(SDK=Path('/CPU_SDK'),IMAGE='CPU_IMAGE',compile_argv=lambda:['FLAG','/leaf/native_rms_rsqrt37_gpu_v1.cpp','LIB'])
  with patch.object(q,'modules',return_value=[p]):argv,compile,runtime=q.expected_recipes(Path('/CPU_OUTPUT'),Path('/CPU_FIXTURE'),123)
  self.assertEqual(argv,['FLAG','/leaf/native_rms_rsqrt37_device_ops_entry_v1.cpp','LIB']);self.assertIn('/leaf/native_rms_rsqrt37_device_ops_entry_v1.cpp',compile[-1]);self.assertIn('SYCL_CACHE_PERSISTENT=0',runtime);self.assertIn(q.SETVARS_PREFIX,runtime[-1])
 def test_combined_comparison_all_device_candidates_and_legacy_hypotheses(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out,old=fields.Tests().fixture(root);oldparent=root/'prior';(oldparent/'build').mkdir(parents=True);old.rename(oldparent/'build/native-output');argument=out/'r0/separate_argument_shadow.f32';item={'path':str(argument),'sha256':q.sha(argument)};fixture={'record':{'hypotheses':{'CPU_FAMILY':{'argument':item,'sqrtf_recip':item,'rounded':item,'legacy':item}}}};r=q.compare_all_variants(out,fixture,{'root':str(oldparent)});self.assertEqual(len(r['device_operation_comparison']['comparisons']),3);self.assertEqual(len(r['all_preregistered_comparisons']['CPU_FAMILY']),4);self.assertIsNone(r['tolerance_gate'])
 def test_inside_binding_dispatch_stays_standalone_stdlib(self):
  s=Path(q.__file__).read_text();tree=ast.parse(s)
  for n in tree.body:
   if isinstance(n,ast.ImportFrom):self.assertEqual(n.module,'pathlib')
  self.assertLess(s.index('if args.inside_runtime:'),s.index('header(args.prior_rms_root)'));self.assertIn("admit_prior(prior_proof['root'])==prior_proof",s);self.assertIn("admit_prior(args.prior_rms_root)==prior_proof",s)
 def test_frozen_preregistered_proposal_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-device-ops-source-plan-v1.json')),'6543d41674c7c39471d2356bc20827a0e9d15c4d0c104c30fb38530b66313aea')
if __name__=='__main__':unittest.main()
