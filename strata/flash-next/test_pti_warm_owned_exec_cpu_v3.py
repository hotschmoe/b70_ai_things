import copy,json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pti_warm_owned_exec_v3 as p
class Proc:
 returncode=0;stdout=None;stderr=None;stdin=None
 def communicate(self,timeout=None):return b'actual metadata stdout',None
 def poll(self):return 0
 def wait(self):return 0
class Tests(unittest.TestCase):
 def owner(self,root=None):return {'Id':'a'*64,'State':{'Running':True,'Pid':123},'HostConfig':{'Binds':[str(p.HERE/'pti_owned_temporal_control_v2.py')+':/pti-temporal.py:ro',(root or p.sampled.EXPECTED_HOST_ROOT)+':/pti:ro']}}
 def test_exact_new_root_and_old_root_refusal(self):
  p.mount_predicate(self.owner())
  with self.assertRaises(ValueError):p.mount_predicate(self.owner('/mnt/vm_8tb/b70/build/pti-unitrace-sampled-query-c388-build-v1/install'))
  obj=self.owner();obj['HostConfig']['Binds'][0]=obj['HostConfig']['Binds'][0].replace(':ro',':rw')
  with self.assertRaises(ValueError):p.mount_predicate(obj)
 def test_actual_invoked_private_run_uses_new_root(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);phase='temporal-before-warmup';raw=struct.pack('<i',1)+bytes(1020);snap={'state':1,'path':'/dev/shm/owned','stat':[45,1,1024,1,1],'uid':1000,'mode':384,'raw_hex':raw.hex()};receipt={'schema':1,'passed':True,'error':None,'session':'owned','action':'observe-paused','target':{'pid':1,'start_ticks':1},'GPU_queries_added':False,'process_memory_read':False,'before':snap,'after':snap};(root/(phase+'.json')).write_text(json.dumps(receipt))
   with patch.object(p.old.lifecycle,'inspect_owned',return_value=self.owner()),patch.object(p.old.subprocess,'Popen',return_value=Proc())as spawn:
    result=p.run(root,'owned',[],'image','binding',phase,'b'*64,'owned')
   self.assertTrue(result['passed']);self.assertEqual(spawn.call_args.args[0],p.command('a'*64,phase,'b'*64,'owned'));self.assertTrue(json.loads((root/(phase+'-exec.receipt.json')).read_bytes())['actual_exec_terminal'])
if __name__=='__main__':unittest.main()
