"""Actual tiny own CPU subprocess retirement and kernel output caps; no model/GPU."""
import os,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import logical_free_worker_owned_v2 as owned
import logical_free_immutable_epoch_v2 as narrow
class Controls(unittest.TestCase):
 def command(self,code):return [str(Path(sys.executable).resolve()),'-I','-B','-S','-c',code]
 def test_all_interrupt_and_error_paths_close_owned_process_and_descriptors(self):
  for error in (KeyboardInterrupt('test interrupt'),RuntimeError('test error'),OSError('test OS error'),subprocess.TimeoutExpired(['CPU'],.1)):
   with self.subTest(error=type(error).__name__),tempfile.TemporaryDirectory()as d:
    base=Path(d);children=[];original=owned.subprocess.Popen
    def create(*args,**kwargs):
     child=original(*args,**kwargs);children.append(child);return child
    prior={sig:signal.getsignal(sig)for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    with patch.object(owned.subprocess,'Popen',side_effect=create),patch.object(original,'communicate',side_effect=error):self.assertRaises(type(error),owned.execute,self.command('import time; time.sleep(60)'),b'',base/'stdout',base/'stderr',base)
    self.assertEqual(len(children),1);child=children[0];self.assertIsNotNone(child.poll());self.assertEqual(owned.session_rows(child.pid),[]);self.assertTrue(child.stdin.closed)
    self.assertEqual({sig:signal.getsignal(sig)for sig in prior},prior)
 def test_actual_timeout_retires_before_exception_returns(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);children=[];original=owned.subprocess.Popen
   def create(*args,**kwargs):
    child=original(*args,**kwargs);children.append(child);return child
   with patch.object(owned.subprocess,'Popen',side_effect=create):self.assertRaises(subprocess.TimeoutExpired,owned.execute,self.command('import time; time.sleep(60)'),b'',base/'stdout',base/'stderr',base,timeout=.01)
   self.assertEqual(owned.session_rows(children[0].pid),[]);self.assertTrue(children[0].stdin.closed)
 def test_kernel_limit_bounds_file_while_child_writes(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);command=self.command('import resource,os; resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096)); os.write(1,b"x"*65536); os.write(1,b"x")');row=owned.execute(command,b'',base/'stdout',base/'stderr',base)
   self.assertNotEqual(row['return_code'],0);self.assertLessEqual((base/'stdout').stat().st_size,4096);self.assertLessEqual((base/'stderr').stat().st_size,4096);self.assertTrue(row['owned_session_empty']);self.assertTrue(row['stdout_stderr_regular_sinks_closed']);self.assertFalse(row['stdout_stderr_pipe_drain_claimed'])
 def test_actual_frozen_worker_reports_cap_and_exact_start_identity(self):
  value,row=narrow.invoke({'mode':'probe','parser_source_base64':__import__('base64').b64encode(narrow.fixed_source()).decode('ascii')});self.assertEqual(value['runtime']['output_file_limit_bytes'],32<<20);self.assertEqual(value['process']['start_ticks'],row['worker_retirement']['worker_start_ticks']);self.assertTrue(row['worker_retirement']['owned_session_empty'])
 def test_cleanup_retry_after_second_interrupt_still_completes(self):
  with tempfile.TemporaryDirectory()as d:
   base=Path(d);original=owned.retire;calls=[]
   def retire(child,start):
    calls.append(1)
    if len(calls)==1:raise KeyboardInterrupt('second cleanup interruption')
    return original(child,start)
   with patch.object(owned,'retire',side_effect=retire):self.assertRaises(subprocess.TimeoutExpired,owned.execute,self.command('import time; time.sleep(60)'),b'',base/'stdout',base/'stderr',base,timeout=.01)
   self.assertGreaterEqual(len(calls),2)
 def test_foreign_PID_session_reuse_is_never_killed(self):
  from types import SimpleNamespace
  child=SimpleNamespace(pid=999,returncode=0,poll=lambda:0,wait=lambda:0)
  with patch.object(owned,'session_rows',return_value=[{'pid':999,'session':999,'start_ticks':123,'state':'S'}]),patch.object(owned.os,'kill',side_effect=AssertionError('foreign process must stay untouched')):
   row=owned.retire(child,122);self.assertTrue(row['foreign_reused_session_observed'])
if __name__=='__main__':unittest.main()
