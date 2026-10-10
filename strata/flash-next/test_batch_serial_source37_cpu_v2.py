"""V1 controls plus actual host stdout EOF/error/retirement negatives, no GPU."""
import copy,io,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from test_batch_serial_source37_cpu_v1 import Serial37Tests,RecoveryTests
import serial37_stdout_capture_v2 as capture

class CaptureTests(unittest.TestCase):
 def test_actual_tiny_process_eof_hash_and_retirement(self):
  with tempfile.TemporaryDirectory(prefix='serial37-stdout-CPU-') as d:
   path=Path(d)/'child-supervisor.log';p=subprocess.Popen([sys.executable,'-c',"print('CPU known line')"],stdout=subprocess.PIPE,text=True);status={'error':None,'eof':False}
   with path.open('w') as log:
    t=threading.Thread(target=capture.forward,args=(p.stdout,log,lambda line:None,status));t.start();self.assertEqual(p.wait(),0);t.join(2);self.assertFalse(t.is_alive())
   p.stdout.close();row=capture.completed(t,status,path);self.assertTrue(row['passed']);self.assertEqual(path.read_text(),'CPU known line\n')
   parent={'child_stdout_capture':row,'child_started_epoch':status['started_epoch']-1,'leaf_logs_epoch':row['finished_epoch']+1,'post_health_finished_epoch':row['finished_epoch']+2};capture.binding(path.parent,parent)
   for key,value in [('eof',False),('error','read error'),('reader_retired',False),('sha256','0'*64)]:
    bad=copy.deepcopy(parent);bad['child_stdout_capture'][key]=value
    with self.assertRaises(ValueError):capture.binding(path.parent,bad)
   parent['child_stdout_capture']['completed_epoch']=parent['leaf_logs_epoch']+2
   with self.assertRaises(ValueError):capture.binding(path.parent,parent)
 def test_stream_error_never_eof_qualified(self):
  class Bad:
   def __iter__(self):raise OSError('synthetic stream error')
  status={'error':None,'eof':False};capture.forward(Bad(),io.StringIO(),lambda line:None,status);self.assertFalse(status['eof']);self.assertIn('OSError',status['error'])
 def test_write_or_forward_error_never_eof_qualified(self):
  for mode in ('write','emit'):
   status={'error':None,'eof':False};log=io.StringIO()
   if mode=='write':log.close()
   def emit(line):
    if mode=='emit':raise OSError('synthetic output error')
   capture.forward(io.StringIO('line\n'),log,emit,status);self.assertIsNotNone(status['error']);self.assertFalse(status['eof'])
 def test_missing_or_live_reader_rejected(self):
  with tempfile.TemporaryDirectory(prefix='serial37-retired-CPU-') as d:
   path=Path(d)/'log';path.write_text('line');status={'started_epoch':time.time(),'completed_epoch':time.time(),'error':None,'eof':True}
   self.assertFalse(capture.completed(None,status,path)['passed']);gate=threading.Event();t=threading.Thread(target=lambda:gate.wait(2));t.start()
   try:self.assertFalse(capture.completed(t,status,path)['passed'])
   finally:gate.set();t.join(2)
 def test_v2_reader_raw_file_join_and_post_recollection_recheck(self):
  s=Path(capture.__file__).with_name('validate_batch_serial_source37_v2.py').read_text();self.assertIn("Individual serial producer raw differs from request row",s);self.assertIn('parent_after,plan_after,child_after=parent_arm(root)',s);self.assertIn('origin_arm(collector);require(parent_after==parent',s)
 def test_v2_real_parent_capture_and_runtime_binding(self):
  h=Path(capture.__file__).parent;s=(h/'qualify_batch_serial_source37_v2.py').read_text();self.assertIn("'parent_generation':42",s);self.assertIn("parent['child_stdout_capture']=completed_stdout",s);self.assertIn('capture_stdout(child.stdout,log',s)
  self.assertIn('stdout_binding(root,parent)',(h/'serial37_runtime_evidence_v2.py').read_text())
if __name__=='__main__':unittest.main()
