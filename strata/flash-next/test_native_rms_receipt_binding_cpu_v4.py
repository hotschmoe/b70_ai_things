"""Original saved producer receipt joins; CPU metadata and tiny stdout only."""
import tempfile,unittest
from pathlib import Path
import qualify_native_rms_rsqrt37_v4 as q
class Tests(unittest.TestCase):
 def fixture(self,root):
  log=root/'compile.log';cmd=root/'compile.command.json';receipt=root/'compile.receipt.json';log.write_text('CPU_PRODUCER_TERMINAL\n',encoding='ascii');q.write(cmd,['CPU_ONLY'])
  row=dict(passed=True,reader_retired=True,eof=True,error=None,command_error=None,phase_cleanup_error=None,return_code=0,path=str(log),sha256=q.sha(log),command=['CPU_ONLY'],command_sha256=q.sha(cmd),started_command_epoch=1.,started_epoch=1.1,completed_epoch=2.,finished_epoch=2.1);q.write(receipt,row);return row,log,cmd,receipt
 def test_actual_shaped_original_saved_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   row,*paths=self.fixture(Path(t));self.assertEqual(q.command_binding(row,*paths),row)
 def test_report_cannot_contradict_original_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   row,*paths=self.fixture(Path(t))
   for key,value in [('return_code',1),('error','RuntimeError'),('command_error','TimeoutExpired'),('phase_cleanup_error','Foreign image'),('started_command_epoch',.5),('started_epoch',1.2),('completed_epoch',2.2),('finished_epoch',2.3),('return_code',False),('started_command_epoch',1)]:
    bad=dict(row);bad[key]=value
    with self.assertRaises(ValueError):q.command_binding(bad,*paths)
 def test_original_receipt_cannot_be_replaced_by_report(self):
  with tempfile.TemporaryDirectory() as t:
   row,*paths=self.fixture(Path(t));bad=dict(row);bad['finished_epoch']=3.;q.write(paths[-1],bad)
   with self.assertRaises(ValueError):q.command_binding(row,*paths)
 def test_matching_phase_cleanup_error_still_fails(self):
  with tempfile.TemporaryDirectory() as t:
   row,*paths=self.fixture(Path(t));row['phase_cleanup_error']='Failed owned removal';q.write(paths[-1],row)
   with self.assertRaises(ValueError):q.command_binding(row,*paths)
 def test_duplicate_receipt_key_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   row,*paths=self.fixture(Path(t));paths[-1].write_text('{"passed":true,"passed":true}\n',encoding='ascii')
   with self.assertRaises(ValueError):q.command_binding(row,*paths)
 def test_all_command_receipts_joined_and_saved_before_verdict(self):
  s=Path(q.__file__).read_text();self.assertIn("label+'.receipt.json' in r['artifact_sha256']",s);self.assertIn("root/(label+'.receipt.json')",s);self.assertLess(s.index("write(out/(label+'.receipt.json'),row)"),s.index("require(row['passed'],'Actual command deadline"))
 def test_previous_frozen_generation_unchanged(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-owned-source-plan-v3.json')),'62f161aa3e7e9db3a8ee35773c2ede4508b74b439abd54ac5a75f0ab628b7a29')
if __name__=='__main__':unittest.main()
