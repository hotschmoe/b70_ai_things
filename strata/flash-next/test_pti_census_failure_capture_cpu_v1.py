"""Real tiny process nonzero/interrupt/error output retained, no Docker/GPU."""
import hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pti_census_owned_exec_v3 as q
class Controls(unittest.TestCase):
 def owner(self):return {'Id':'CPU_CONTAINER','State':{'Running':True,'Pid':77},'HostConfig':{'Binds':[str(q.HERE/'pti_mapped_injection_census_v1.py')+':/pti-census.py:ro','/CPU_INSTALL:/pti:ro']}}
 def run_fixture(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);owner=self.owner();real=subprocess.Popen;seen=[]
   def launch(cmd,**kw):
    seen.append(cmd);proc=real(['/usr/bin/python3','-c','import sys; print("PermissionError: CPU_SYNTHETIC proc/maps refusal",flush=True); sys.exit(7)'],**kw)
    if kind=='interrupt':
     original=proc.communicate;calls=[0]
     def communicate(*a,**k):
      calls[0]+=1
      if calls[0]==1:raise InterruptedError('CPU after actual launch')
      return original(*a,**k)
     proc.communicate=communicate
    return proc
   observations=[owner,owner]if kind!='owner_after'else[owner,ValueError('CPU owner inspect failed')]
   with patch.object(q.lifecycle,'inspect_owned',side_effect=observations),patch.object(q.subprocess,'Popen',side_effect=launch):
    with self.assertRaises(ValueError):q.run(root,'CPU_NATIVE',['CPU_RECIPE'],'CPU_IMAGE','a'*64,'after_READY','b'*64)
   diag=json.loads((root/'after_READY-exec.receipt.json').read_text());data=(root/'after_READY-exec.stdout').read_bytes();self.assertIn(b'PermissionError:',data);self.assertEqual(diag['return_code'],7);self.assertFalse(diag['passed']);self.assertTrue(diag['actual_exec_terminal']);self.assertEqual(diag['metadata_stdout_sha256'],hashlib.sha256(data).hexdigest());self.assertEqual(diag['command'],seen[0]);self.assertEqual(json.loads((root/'after_READY-exec.command.json').read_text()),seen[0]);self.assertTrue(diag['errors'])
   if kind!='owner_after':self.assertTrue((root/'after_READY-owner-after.json').is_file())
 def test_nonzero_real_process_output_and_terminal_retained(self):self.run_fixture('nonzero')
 def test_interrupted_real_process_still_drains_output_and_fails(self):self.run_fixture('interrupt')
 def test_owner_after_failure_does_not_discard_exec_output(self):self.run_fixture('owner_after')
if __name__=='__main__':unittest.main()
