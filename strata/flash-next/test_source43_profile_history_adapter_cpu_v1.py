import copy,unittest
from pathlib import Path
from unittest.mock import patch
import source43_profile_normal_history_adapter_v1 as a
class Controls(unittest.TestCase):
 def test_named_root_and_false_scope(self):
  value={'historical_completed_evidence_only':True,'current_model_identity_qualified':False}
  with patch.object(a.history,'binding',return_value=value):self.assertIs(a.prerequisite(a.history.NORMAL),value);self.assertRaises(ValueError,a.prerequisite,Path('/foreign'))
 def test_actual_current_admission_called_unchanged(self):
  plan={'source43':{'actual':'current'}};metadata={'SDK':'admitted'};cfg={'args':['unchanged']};seen=[]
  def original(p):seen.append(p);return metadata,cfg
  with patch.object(a.history,'binding',return_value={'historical':True}),patch.object(a.history,'match_current_contract',return_value={'current_SDK_bytes_verified_by_this_function':False,'fresh_per_arm_identity_and_runtime_gates_still_required':True})as matched:
   result=a.admission(plan,original);self.assertIs(result[0],metadata);self.assertIs(result[1],cfg);self.assertEqual(seen,[plan]);matched.assert_called_once_with({'historical':True},plan['source43'],cfg)
 def test_original_current_failure_not_waived(self):
  with patch.object(a.history,'binding',side_effect=AssertionError('No history substitution after current failure')):self.assertRaisesRegex(ValueError,'current SDK failed',a.admission,{},lambda _ :(_ for _ in()).throw(ValueError('current SDK failed')))
if __name__=='__main__':unittest.main()
