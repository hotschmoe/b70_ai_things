import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_owned_container_logs_v1 as a
class Logs(unittest.TestCase):
 def write(self,p,v):p.write_text(json.dumps(v,sort_keys=True)+'\n')
 def obj(self):return {'Id':'a'*64,'State':{'Running':False,'ExitCode':1,'OOMKilled':False,'Error':''}}
 def capture(self,root,script,cap=1000):
  obj=self.obj();self.write(root/'owned-terminal-inspection.json',obj)
  def launch(command,**kw):self.assertEqual(command,['docker','logs','--timestamps','a'*64]);return subprocess.Popen([sys.executable,'-c',script],**kw)
  return a.capture(root,obj,self.write,popen=launch,max_bytes=cap,timeout=5)
 def test_real_tiny_stdout_stderr_receipt_and_mutation(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);row=self.capture(root,"import os;os.write(1,b'original\\x00stdout');os.write(2,b'original stderr\\n')");self.assertTrue(row['complete']);a.binding(root,self.obj(),row)
   (root/'owned-container-stderr.raw').write_bytes(b'changed')
   with self.assertRaises(ValueError):a.binding(root,self.obj(),row)
 def test_oversize_is_explicit_failed_capture_never_normal(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);row=self.capture(root,"import os;os.write(1,b'x'*100)",10);self.assertFalse(row['complete']);self.assertIsNotNone(row['error']);self.assertLessEqual((root/'owned-container-stdout.raw').stat().st_size,10)
   with self.assertRaises(ValueError):a.binding(root,self.obj(),row)
 def test_actual_launch_exception_records_failure_and_preserves_empty_streams(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);obj=self.obj();self.write(root/'owned-terminal-inspection.json',obj)
   def launch(*args,**kw):raise OSError('original launch failure')
   row=a.capture(root,obj,self.write,popen=launch,max_bytes=20,timeout=5);self.assertFalse(row['complete']);self.assertIsNone(row['return_code']);a.binding(root,obj,row,require_complete=False)
   with self.assertRaises(ValueError):a.binding(root,obj,row)
 def test_wrong_inspected_id_and_live_owner_refused_before_launch(self):
  with tempfile.TemporaryDirectory()as temp:
   for value in ('foreign',True):
    obj=self.obj();obj['Id']=value
    with self.assertRaises(ValueError):a.capture(temp,obj,self.write,popen=lambda *a,**k:self.fail('launched'))
   obj=self.obj();obj['State']['Running']=True
   with self.assertRaises(ValueError):a.capture(temp,obj,self.write,popen=lambda *a,**k:self.fail('launched'))
 def test_actual_retire_capture_precedes_rm_and_receipt_is_reused(self):
  import full_cache_shared_actor_retirement_v9 as retire
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);events=[]
   def capture(root,obj,write):events.append('logs');return {'complete':True}
   def run(command,**kw):events.append(command);return type('Result',(),{'returncode':0})()
   with patch.object(a,'capture',capture):result=retire.retire('owned',self.obj,run,lambda name:True,lambda t:None,self.write,root)
   self.assertEqual(events,['logs',['docker','rm','owned']]);self.assertFalse(result['normal_terminal'])
if __name__=='__main__':unittest.main()
