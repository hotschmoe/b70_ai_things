"""Actual-shaped nested receipt negatives without model or actor execution."""
import tempfile,unittest
from pathlib import Path
import validate_full_cache_shared_suite_v3 as v

class Reader(unittest.TestCase):
 def test_actual_parent_epoch_bracket_and_raw_stdout(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'stdout.log';path.write_text('normal EOF\n');receipt={'command':['python','parent'],'producer_pid':17,'producer_identity':{'pid':17,'start_ticks':123},'identity_error':None,'return_code':0,'interruption_signals':[],'actual_child_terminal_and_file_EOF':True,'started_epoch':1.0,'finished_epoch':5.0,'stdout_sha256':v.ctrl.sha(path)};parent={'started_epoch':2.0,'finished_epoch':4.0,'parent_pid':17,'parent_start_ticks':123}
   self.assertTrue(v.receipt_binding(receipt,receipt['command'],path,parent))
   receipt['producer_pid']=999
   with self.assertRaises(ValueError):v.receipt_binding(receipt,receipt['command'],path,parent)
   receipt['producer_pid']=17;parent['finished_epoch']=6.0
   with self.assertRaises(ValueError):v.receipt_binding(receipt,receipt['command'],path,parent)
   parent['finished_epoch']=4.0;path.write_text('truncated')
   with self.assertRaises(ValueError):v.receipt_binding(receipt,receipt['command'],path,parent)
 def test_interruption_or_boolean_rc_never_normal_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'log';path.write_text('EOF\n');receipt={'command':['parent'],'producer_pid':17,'producer_identity':{'pid':17,'start_ticks':123},'identity_error':None,'return_code':False,'interruption_signals':[],'actual_child_terminal_and_file_EOF':True,'started_epoch':1,'finished_epoch':4,'stdout_sha256':v.ctrl.sha(path)}
   with self.assertRaises(ValueError):v.receipt_binding(receipt,['parent'],path,{'started_epoch':2,'finished_epoch':3,'parent_pid':17,'parent_start_ticks':123})
   receipt['return_code']=0;receipt['interruption_signals']=[15]
   with self.assertRaises(ValueError):v.receipt_binding(receipt,['parent'],path,{'started_epoch':2,'finished_epoch':3,'parent_pid':17,'parent_start_ticks':123})
if __name__=='__main__':unittest.main()
