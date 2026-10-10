import os,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_qsa40_byte_epoch_v5 as e
import native_qsa40_owned_launch_v5 as launch
from native_qsa40_protocol_v5 import Protocol
class Controls(unittest.TestCase):
 def setup_tree(self,root):
  data=root/'data';data.mkdir();(data/'a').write_bytes(b'valid');models=[]
  for i in range(4):p=root/('model'+str(i));p.write_bytes(b'M');models.append(p)
  return {k:str(data)for k in('prepared','engine_root','pack','own_producer_root')},data,models
 def test_symlink_path_refused_before_any_full_read(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,data,models=self.setup_tree(root);link=root/'alias';link.symlink_to(data,target_is_directory=True);plan['prepared']=str(link)
   with patch.object(e,'current_row')as read:self.assertRaises(ValueError,e.ByteEpoch,plan,[],models);read.assert_not_called()
 def test_sparse_perfile_and_total_bounds_precede_whole_reads(self):
  for single,total in[(4,100),(100,4)]:
   with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp);plan,data,models=self.setup_tree(root)
    with patch.object(e,'MAX_FILE_BYTES',single),patch.object(e,'MAX_TOTAL_BYTES',total),patch.object(e,'current_row')as read:
     self.assertRaises(ValueError,e.ByteEpoch,plan,[],models);read.assert_not_called()
 def test_explicit_plan_only_allowed_root_is_in_complete_traversal(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,data,models=self.setup_tree(root);foreign=root/'foreign';foreign.mkdir();(foreign/'private').write_bytes(b'not_declared');plan['candidate_binding']={'foreign_root':str(foreign)}
   with patch.object(e,'PREFIXES',(root,)):
    roots,files=e.roots_and_files(plan,[]);self.assertIn(foreign,roots)
 def test_unauthenticated_published_parent_refused_before_whole_read(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,data,models=self.setup_tree(root);(data/'prepared.json').write_text('{}');plan['prepared_sha256']='0'*64
   with patch.object(e,'current_row')as read:self.assertRaises(ValueError,e.ByteEpoch,plan,[],models);read.assert_not_called()
 def test_initial_absence_not_a_retired_original_launch(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);launch.intent(root,['CPU_NO_GPU']);self.assertFalse(launch.fence(root));launch.write(root/'launch-failure.json',{'command':['CPU_NO_GPU'],'actual_client_started':False,'error':'synthetic Popen refused'});self.assertTrue(launch.fence(root))
 def test_deferred_signal_does_not_interrupt_protected_launch_bookkeeping(self):
  events=[];old=signal.getsignal(signal.SIGTERM)
  with self.assertRaises(InterruptedError):
   with launch.deferred_signals():events.append('protected');os.kill(os.getpid(),signal.SIGTERM);events.append('recorded_original_launch')
  self.assertEqual(events,['protected','recorded_original_launch']);self.assertEqual(signal.getsignal(signal.SIGTERM),old)
 def test_Popen_refusal_restores_handlers_and_fences_no_started_client(self):
  old=signal.getsignal(signal.SIGTERM)
  with tempfile.TemporaryDirectory()as tmp,patch('native_qsa40_protocol_v5.subprocess.Popen',side_effect=OSError('CPU launch refused')):
   out=Path(tmp);self.assertRaises(OSError,Protocol,['CPU_NO_GPU'],out,lambda:None);self.assertTrue(launch.fence(out));self.assertEqual(signal.getsignal(signal.SIGTERM),old)
 def test_actual_immediate_exit_original_client_is_retired(self):
  import sys
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);self.assertRaises(ValueError,Protocol,[sys.executable,'-c','pass'],out,lambda:None,ready_timeout=5);self.assertTrue(launch.fence(out));record=launch.read_unique(out/'launch.json');self.assertTrue(record['actual_client_started']);self.assertEqual(launch.read_unique(out/'launch-retired.json')['return_code'],0)
 def test_actual_started_immediate_client_missing_proc_identity_still_retires(self):
  import sys
  actual=launch.identity
  def identity(pid):
   if pid!=os.getpid():raise FileNotFoundError('synthetic immediate original-client /proc disappearance')
   return actual(pid)
  with tempfile.TemporaryDirectory()as tmp,patch.object(launch,'identity',side_effect=identity):
   out=Path(tmp);self.assertRaises(ValueError,Protocol,[sys.executable,'-c','pass'],out,lambda:None,ready_timeout=5);self.assertTrue(launch.fence(out));record=launch.read_unique(out/'launch.json');self.assertTrue(record['actual_client_started']);self.assertFalse(record['client_identity_observed']);self.assertIsNone(record['client']['start_ticks']);self.assertEqual(launch.read_unique(out/'launch-retired.json')['return_code'],0)
 def test_original_READY_parent_exact_start_identity(self):
  from native_qsa40_evidence_v5 import parent_identity
  report={'producer_pid':123,'producer_identity':{'pid':123,'start_ticks':456}};ready={'parent':{'pid':123,'start_ticks':456}};self.assertTrue(parent_identity(report,ready))
  for row in ({'pid':123,'start_ticks':457},{'pid':124,'start_ticks':456},{'pid':True,'start_ticks':456}):self.assertRaises(ValueError,parent_identity,report,{'parent':row})
if __name__=='__main__':unittest.main()
