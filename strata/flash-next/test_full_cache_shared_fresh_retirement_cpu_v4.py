"""Tiny handles/EOF/retirement tests; no Docker or model launch."""
import io,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_fresh_retirement_v4 as f

class Fresh(unittest.TestCase):
 def test_failed_readiness_preserves_original_created_handle(self):
  class Broken:
   def __init__(self,*args):self.p=object();raise ValueError('readiness failed')
  protocol=f.original_protocol(Broken,[],None)
  with self.assertRaises(ValueError):Broken.__init__(protocol)
  self.assertTrue(hasattr(protocol,'p'))
 def test_original_wait_and_pump_join_before_critical_marker_clears(self):
  events=[]
  class Process:
   stdin=io.StringIO();stdout=io.StringIO();stderr=io.StringIO()
   def poll(self):return 0
   def wait(self):events.append('wait');return 0
  protocol=types.SimpleNamespace(p=Process(),threads=[types.SimpleNamespace(join=lambda:events.append('pumpEOF'))],err=io.StringIO(),stdout=['READY'])
  def settled(*args):return {'normal_terminal':True},{'interruption_signals':[],'failure_count':0,'retirement_in_progress':False}
  def write(path,value):events.append(('ledger',value['retirement_in_progress']))
  with tempfile.TemporaryDirectory() as tmp,patch.object(f,'retain_until_settled',side_effect=settled):
   rc,result,ledger,errors=f.finish(protocol,'owned',None,None,None,write,Path(tmp))
  self.assertEqual(rc,0);self.assertEqual(errors,[]);self.assertTrue(ledger['original_protocol_CLI_and_pipe_EOF_joined']);self.assertLess(events.index('wait'),events.index(('ledger',False)));self.assertLess(events.index('pumpEOF'),events.index(('ledger',False)))
 def test_unobserved_original_launch_never_normal_closure(self):
  def settled(*args):return {'normal_terminal':True},{'interruption_signals':[],'failure_count':0,'retirement_in_progress':False}
  with tempfile.TemporaryDirectory() as tmp,patch.object(f,'retain_until_settled',side_effect=settled):
   rc,result,ledger,errors=f.finish(None,'owned',None,None,None,lambda *args:None,Path(tmp))
  self.assertIsNone(rc);self.assertTrue(errors)
 def test_actual_tiny_SIGTERM_enters_controlled_finally(self):
  import os,subprocess,sys
  script="""import os,signal
import full_cache_shared_fresh_retirement_v4 as f
row={};previous=f.controlled_signals(row);entered=False
try:
 try:os.kill(os.getpid(),signal.SIGTERM)
 except KeyboardInterrupt:entered=True
 finally:assert row['interruption_signals']==[signal.SIGTERM]
 assert entered
finally:f.restore_signals(previous)
"""
  result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=5,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(Path(f.__file__).parent)})
  self.assertEqual(result.returncode,0,result.stderr)
if __name__=='__main__':unittest.main()
