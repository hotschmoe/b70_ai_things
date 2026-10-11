import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import admit_source41_model_fidelity_v1 as a
class Controls(unittest.TestCase):
 def fixture(self,root):
  captured=root/'actor/capture';captured.mkdir(parents=True);(captured/'requests.json').write_text('[]\n');reference=root/'original';reference.mkdir()
  descriptor={'capture_root':str(captured),'materialized_mode':1,'observer_enabled':True,'source41_sdk':{'actual_source':41},'engine_pid':123,'binding_sha256':'a'*64}
  return captured.parent,reference,descriptor
 def run_fixture(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,reference,before=self.fixture(root);after=copy.deepcopy(before)
   if kind=='changed_after':after['engine_pid']=999
   if kind=='bad_mode':before['materialized_mode']=True;after=copy.deepcopy(before)
   failure=ValueError('reference read failed')if kind=='reference_failure'else None
   with patch.object(a,'source_binding',return_value='frozen'),patch.object(a.actor,'captured_binding',side_effect=[before,after])as admission,patch.object(a.raw,'captured_arm',return_value={'actual':'raw'})as raw,patch.object(a.original,'join',side_effect=failure,return_value={'full_model_math_qualified':False,'comparisons':'diagnostic'})as original:
    output=root/'external.json'
    if kind=='valid':
     result=a.finalized_binding(actor,reference,output);self.assertTrue(result['passed']);self.assertFalse(result['full_model_math_qualified']);self.assertFalse(result['tolerance_fitted'])
    else:
     with self.assertRaises(ValueError):a.finalized_binding(actor,reference,output)
     self.assertFalse(json.loads(output.read_text())['passed'])
    self.assertEqual(admission.call_count,2)
    if kind=='bad_mode':self.assertFalse(raw.called);self.assertFalse(original.called)
 def test_actual_enclosing_order_and_diagnostic_scope(self):self.run_fixture('valid')
 def test_descriptor_mutation_after_comparison_refused(self):self.run_fixture('changed_after')
 def test_failure_after_reference_read_still_admits_original_actor(self):self.run_fixture('reference_failure')
 def test_bool_mode_refused_before_raw_or_original(self):self.run_fixture('bad_mode')
if __name__=='__main__':unittest.main()
