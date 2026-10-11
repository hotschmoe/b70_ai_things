import os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import qualify_indexer_materialized43_masked_upload_view_v2 as q

class Controls(unittest.TestCase):
 def fixture(self,root,baseline_failure=False,double=False,predicate_failure=False):
  (root/'report.json').write_text('{}');upload={'actual_source390_upload_qualified':True,'component_fixture':True};proof={'model_baseline_or_constructor_proof_transferred':False,'actual_fresh43_upload_binding':upload};component={'report_sha256':'original','comparison':{'actual_fixture':True},'full_model_math_qualified':False}
  baseline=type('Baseline',(),{});baseline.finalized_binding=Mock(side_effect=ValueError('actual upload failed')if baseline_failure else None,return_value=({'actual41':True},proof))
  modules=lambda:(object(),object(),object(),object(),object(),baseline)
  namespace={'modules':modules,'component':component,'double':double,'predicate_failure':predicate_failure}
  exec('def finalized_binding(root):\n base=modules()[-1]\n base.finalized_binding(root,None)\n if double:base.finalized_binding(root,None)\n if predicate_failure:raise ValueError("original predicate failed")\n return component',namespace)
  return namespace,baseline,upload,component,modules
 def run_fixture(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);namespace,baseline,upload,component,modules=self.fixture(root,kind=='baseline_failure',kind=='double',kind=='predicate_failure');original_modules=q.original.modules;original_baseline=q.original.Baseline
   witness_admit=Mock(side_effect=ValueError('original witness failed')if kind=='witness_failure'else None,return_value={'original_witness_admitted':True})
   def witness():return {'owner_pid':os.getpid()+int(kind=='foreign_pid'),'one_full_semantic_admission_per_owned_process':True,'whole_current_input_bytes_on_each_use':True,'generic_persistent_cached_PASS_used':False,'boundaries':[{'owner_pid':os.getpid(),'finished_epoch':time.time()}]}
   with patch.object(q,'source_binding',return_value='source'),patch.object(q.original,'derived',return_value=namespace),patch.object(q.inputs,'admit_operation_witness',witness_admit),patch.object(q.inputs,'operation_witness',side_effect=witness):
    if kind=='valid':
     result=q.finalized_binding(root);self.assertIs(result['original_masked_component_binding'],component);self.assertIs(result['actual_revalidated_upload_binding'],upload);self.assertEqual(result['same_call']['original_Baseline_calls'],1);self.assertFalse(result['same_call']['saved_PASS_substituted']);witness_admit.assert_called_once()
    else:
     with self.assertRaises(ValueError):q.finalized_binding(root)
   self.assertEqual(baseline.finalized_binding.call_count,1);self.assertIs(q.original.modules,original_modules);self.assertIs(q.original.Baseline,original_baseline);self.assertIs(namespace['component'],component)
 def test_actual_predicate_one_upload_return_exact_and_no_global_mutation(self):self.run_fixture('valid')
 def test_actual_upload_failure_propagates(self):self.run_fixture('baseline_failure')
 def test_duplicate_baseline_call_refused_before_second_upload(self):self.run_fixture('double')
 def test_original_predicate_failure_not_replaced_by_saved_pass(self):self.run_fixture('predicate_failure')
 def test_original_operation_witness_failure_propagates(self):self.run_fixture('witness_failure')
 def test_foreign_pid_live_witness_refused(self):self.run_fixture('foreign_pid')
if __name__=='__main__':unittest.main()
