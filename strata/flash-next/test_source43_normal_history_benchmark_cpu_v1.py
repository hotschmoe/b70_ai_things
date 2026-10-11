import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_source43_normal_history_v1 as b
class Controls(unittest.TestCase):
 def fixture(self):
  public={'passed':True,'current_public_full4_rehashed':True};association={'original_public_admission':public,'current_model_identity_qualified':False,'current_public_full4_rehashed':False,'current_GPU_health_transferred':False};return public,association
 def test_actual_complete_ABBA_and_result_equal(self):
  public,assoc=self.fixture();calls=[]
  with tempfile.TemporaryDirectory()as td,patch.object(b,'sources',return_value='source'):
   result=b.run(Path(td)/'out','no-payload',1,lambda:(calls.append('A')or copy.deepcopy(public)),lambda:(calls.append('B')or copy.deepcopy(assoc)),lambda _:{'current_source':'fixture'})
   self.assertEqual([r['arm']for r in result['trials']],['A','B','B','A']);self.assertTrue(result['all_original_public_results_equal']);self.assertFalse(result['A_and_B_current_identity_authority_equal'])
 def test_wrong_original_result_and_attempt_retained(self):
  public,assoc=self.fixture()
  with tempfile.TemporaryDirectory()as td,patch.object(b,'sources',return_value='source'):
   out=Path(td)/'out';self.assertRaises(ValueError,b.run,out,'no-payload',1,lambda:{'passed':False},lambda:copy.deepcopy(assoc),lambda _:{});report=json.loads((out/'report.json').read_text());self.assertFalse(report['passed']);self.assertEqual(len(report['trials']),1);self.assertTrue(report['trials'][0]['error'])
 def test_B_cannot_become_current_identity(self):
  public,assoc=self.fixture();attempt=[0]
  def association():
   attempt[0]+=1;value=copy.deepcopy(assoc)
   if attempt[0]>1:value['current_model_identity_qualified']=True
   return value
  with tempfile.TemporaryDirectory()as td,patch.object(b,'sources',return_value='source'):
   self.assertRaises(ValueError,b.run,Path(td)/'out','no-payload',1,lambda:copy.deepcopy(public),association,lambda _:{})
 def test_inherited_GPU_lease_refused_before_original_reader(self):
  with patch.object(b.os,'readlink',return_value='/mnt/vm_8tb/b70/gpu.lock.1'):
   self.assertRaises(ValueError,b.no_inherited_GPU_leases)
if __name__=='__main__':unittest.main()
