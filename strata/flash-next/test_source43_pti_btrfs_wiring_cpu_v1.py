"""Saved source-shaped Btrfs/cmd joins; no actual library/model/Docker access."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source43_pti_latency_v3 as a
import validate_source43_pti_latency_v3 as r
import pti_mapped_injection_census_v2 as helper
import test_source43_pti_latency_cpu_v1 as old_tests
class Controls(unittest.TestCase):
 def fixture(self,root,kind):
  digest,terminal,recipe=old_tests.CensusControls().fixture(root,None);mount='711 1 0:50 '+a.census.EXPECTED_BIND_ROOT+' /pti ro - btrfs /dev/sdc1 rw,subvolid=5\n';maps='1000-2000 r-xp 00000000 00:32 7 /pti/lib/libunitrace_tool.so\n'
  if kind=='foreignroot':mount=mount.replace(a.census.EXPECTED_BIND_ROOT,'/foreign/install')
  if kind=='arbitrarydev':maps=maps.replace('00:32','00:33')
  for phase in('after_READY','after_last_DONE_before_QUIT'):
   directory=root/phase;receipt=json.loads((directory/'receipt.json').read_text());segments=helper.select(maps);cover=helper.covering_mount(mount,helper.LIB);stat=[45,7,len(b'\x7fELFCPU_SYNTHETIC'),2,3];receipt.update(schema=2,backing_file_stat=stat,path_stat_after=stat,opened_fd_stat_before=stat,opened_fd_stat_after=stat,map_file_targets=[helper.LIB],mapped_segments=segments,expected_bind_root=a.census.EXPECTED_BIND_ROOT,expected_filesystem_source=a.census.EXPECTED_FILESYSTEM_SOURCE,covering_mount_before=cover,covering_mount_after=cover)
   if kind is None:receipt['btrfs_device_domain_association']=helper.associate(segments,stat,stat,stat,stat,cover,cover,helper.LIB,a.census.EXPECTED_BIND_ROOT,a.census.EXPECTED_FILESYSTEM_SOURCE,digest,digest,[helper.LIB])
   else:receipt['btrfs_device_domain_association']={}
   (directory/'receipt.json').write_text(json.dumps(receipt));(directory/'proc1.maps').write_text(maps);(directory/'proc1.mountinfo').write_text(mount);(directory/'proc1.mountinfo.after').write_text(mount)
   row=json.loads((root/(phase+'-owned.json')).read_text());row['command']=a.census.command(terminal['Id'],phase,digest);row['census']=receipt;(root/(phase+'-owned.json')).write_text(json.dumps(row));diag={'passed':True,'errors':[],'actual_exec_started':True,'actual_exec_terminal':True,'return_code':0,'command':row['command'],'container_id':terminal['Id'],'metadata_stdout_sha256':__import__('hashlib').sha256(b'').hexdigest(),'metadata_stdout_bytes':0};(root/(phase+'-exec.receipt.json')).write_text(json.dumps(diag))
  return digest,terminal,recipe
 def exercise(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);digest,terminal,recipe=self.fixture(root,kind)
   with patch.object(a.tool,'LIB_SHA',digest),patch.object(a.life.contract,'inspection'):
    if kind:
     with self.assertRaises(ValueError):r.census_binding(root,recipe,terminal,[])
    else:self.assertTrue(r.census_binding(root,recipe,terminal,[])['actual_mapped_injection_before_after_joined'])
 def test_anon_stat45_and_mount_vma50_sameinode_accepted(self):self.exercise(None)
 def test_arbitrary_map_device33_rejected(self):self.exercise('arbitrarydev')
 def test_foreign_install_mountroot_rejected(self):self.exercise('foreignroot')
 def test_actor_and_exec_exact_new_helper_command(self):
  cmd=a.census.command('CID','after_READY','a'*64);self.assertEqual(cmd[-4:],['--expected-bind-root',a.census.EXPECTED_BIND_ROOT,'--expected-filesystem-source','/dev/sdc1']);self.assertTrue(a.census.__file__.endswith('pti_census_owned_exec_v4.py'));self.assertTrue(a.PLAN.name.endswith('v3.json'))
 def test_failure_retention_body_preserved(self):
  h=Path(a.__file__).parent;before=ast.parse((h/'pti_census_owned_exec_v3.py').read_text());after=ast.parse((h/'pti_census_owned_exec_v4.py').read_text());one=next(n for n in before.body if isinstance(n,ast.FunctionDef)and n.name=='run');two=next(n for n in after.body if isinstance(n,ast.FunctionDef)and n.name=='run');self.assertEqual(len([n for n in ast.walk(one)if isinstance(n,ast.Try)]),len([n for n in ast.walk(two)if isinstance(n,ast.Try)]));self.assertIn('-exec.receipt.json',ast.unparse(two));self.assertIn('while proc.poll() is None',ast.unparse(two))
if __name__=='__main__':unittest.main()

class ActualFailureControls(unittest.TestCase):
 def test_failure_capture_remains_actual_terminal_on_new_helper(self):
  import test_pti_census_failure_capture_cpu_v1 as parent
  class NewOwner(parent.Controls):
   def owner(self):return {'Id':'CPU_CONTAINER','State':{'Running':True,'Pid':77},'HostConfig':{'Binds':[str(a.census.HERE/'pti_mapped_injection_census_v2.py')+':/pti-census.py:ro',a.census.EXPECTED_HOST_ROOT+':/pti:ro']}}
  with patch.object(parent,'q',a.census):NewOwner().run_fixture('nonzero')
