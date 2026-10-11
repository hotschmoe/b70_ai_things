"""Negative mocked actor boundary with real local SIGTERM, no native launch."""
import json,os,signal,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import run_full_cache_shared_fresh40_v7 as actor
import run_full_cache_shared_batch0_fresh40_v7 as serial
import full_cache_shared_fresh_retirement_v7 as retirement

class Boundary(unittest.TestCase):
 def check_actor(self,module):
  seen=[];before=signal.getsignal(signal.SIGTERM)
  def initialize(protocol,command,output):
   protocol.p=object();os.kill(os.getpid(),signal.SIGTERM)
  def finish(protocol,*args):
   seen.append(protocol.p);return 0,{'state':{'Running':False,'ExitCode':0,'OOMKilled':False},'removed':True,'normal_terminal':True},{'failure_count':0,'interruption_signals':[]},[]
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);health=root/'health.json';health.write_text(json.dumps({'passed':True,'cards':[0,1],'finished_epoch':module.time.time()}));plan={'cards':[0,1],'jobs':[]};raw=json.dumps(plan).encode()
   with patch.object(module.ctrl,'source_binding',return_value={}),patch.object(module.ctrl.c1,'leased'),patch.object(module.ctrl,'command_recipe',return_value=['docker','run','--name','CPU-negative-never-launched']),patch.object(module.FreshProtocol,'__init__',side_effect=initialize,autospec=True),patch.object(retirement,'finish',side_effect=finish):
    result=module.run(plan,root/'negative',health,raw,{}, {})
   self.assertFalse(result['passed']);self.assertFalse(result['collection_and_teardown_passed']);self.assertEqual(result['interruption_signals'],[signal.SIGTERM]);self.assertIn('KeyboardInterrupt',result['error']);self.assertEqual(len(seen),1);self.assertFalse(result['full_cache_runtime_qualified'])
  self.assertEqual(signal.getsignal(signal.SIGTERM),before)
 def test_fresh_constructor_signal_preserves_handle_into_cleanup(self):self.check_actor(actor)
 def test_serial_fresh_constructor_signal_preserves_handle_into_cleanup(self):self.check_actor(serial)
if __name__=='__main__':unittest.main()
