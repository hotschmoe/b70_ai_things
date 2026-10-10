"""Actual-shaped Docker reorder and exact identity environment controls."""
import copy,unittest
from full_cache_shared_memory_capture_v5 import recipe_binding
from run_full_cache_shared_runtime_v5 import runtime_identity_environment
class Controls(unittest.TestCase):
 def fixture(self):
  binds=['/sdk:/src:ro','/out:/results:rw','/repo:/controller:ro'];command=['docker','run','--user','1000:1000','--network','host','--group-add','107']
  for b in binds:command+=['-v',b]
  command+=['-e','STRATA_TEST=1','image','-lc','exec engine'];obj={'Image':'image','Config':{'Image':'image','Cmd':['-lc','exec engine'],'Entrypoint':['/bin/bash'],'User':'1000:1000','Env':['STRATA_TEST=1']},'HostConfig':{'NetworkMode':'host','Memory':105*1024**3,'MemorySwap':105*1024**3,'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'GroupAdd':['107'],'Binds':list(reversed(binds))},'Mounts':[{'Source':a,'Destination':b,'RW':mode=='rw','Type':'bind'}for a,b,mode in(v.rsplit(':',2)for v in binds)]};return obj,command
 def test_actual_docker_reorders_identical_mounts(self):
  obj,cmd=self.fixture();self.assertTrue(recipe_binding(obj,cmd,'image'))
 def test_missing_duplicate_or_changed_mount_rejected(self):
  obj,cmd=self.fixture()
  for binds in (obj['HostConfig']['Binds'][:-1],obj['HostConfig']['Binds']+[obj['HostConfig']['Binds'][0]],['/foreign:/src:ro',*obj['HostConfig']['Binds'][:-1]]):
   bad=copy.deepcopy(obj);bad['HostConfig']['Binds']=binds;self.assertRaises(ValueError,recipe_binding,bad,cmd,'image')
 def test_full_environment_unchanged_identity_projection_exact(self):
  full={'ZE_AFFINITY_MASK':'0','SYCL_UR_TRACE':'2','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','STRATA_TEST':'1','STRATA_ARTIFACT_IDENTITY_SHA256':'x','PATH':'/bin'};saved=dict(full)
  self.assertEqual(runtime_identity_environment(full),{'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','STRATA_TEST':'1','SYCL_UR_TRACE':'2'});self.assertEqual(full,saved)
 def test_explicit_external_removal_recovers_only_failure(self):
  import json,tempfile
  from pathlib import Path
  from unittest.mock import patch
  from full_cache_shared_actor_retirement_v5 import external_recovery
  obj,cmd=self.fixture();obj.update(Name='/owned',Id='container-id');obj['State']={'Running':False,'ExitCode':2,'OOMKilled':False,'Error':''};obj['Config']['Labels']={'owner':'original'};cmd[2:2]=['--label','owner=original']
  owner={'pid':123,'start_ticks':456,'session':123,'state':'S'}
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary)
   for name,value in [('launch.command.json',cmd),('actor-owner.json',owner),('external-owned-terminal-removal.json',{'schema':1,'actor_owner':owner,'command':cmd,'name':'owned','normal_qualification':False,'terminal_inspection':obj,'container_id':'container-id','removal_receipt':{'command':['docker','rm','container-id'],'return_code':0,'error':None}})]:
    (root/name).write_text(json.dumps(value))
   with patch('full_cache_shared_runtime_v5.c1.leased'):
    result=external_recovery(root,'owned',lambda _:True);self.assertFalse(result['normal_terminal']);self.assertTrue(result['removed'])
    receipt=json.loads((root/'external-owned-terminal-removal.json').read_text());receipt['actor_owner']['pid']=123.0;(root/'external-owned-terminal-removal.json').write_text(json.dumps(receipt));self.assertRaises(ValueError,external_recovery,root,'owned',lambda _:True)
 def test_projection_matches_exact_consumed_source_function(self):
  import ast,hashlib
  from pathlib import Path
  source=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo/source/serve/artifact_identity.py').read_bytes()
  self.assertEqual(hashlib.sha256(source).hexdigest(),'47619fcd5def60d13861773b359e9bab35499212f858ba649ce6462feb15027f')
  tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='runtime_environment');globals={'ENV_PREFIXES':('STRATA_','SYCL_','ONEAPI_'),'IDENTITY_ENV':'STRATA_ARTIFACT_IDENTITY_SHA256'};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'pinned-source-runtime-environment','exec'),globals)
  env={'STRATA_X':0,'SYCL_X':2,'ONEAPI_X':'gpu','ZE_AFFINITY_MASK':'0','PATH':'/bin','STRATA_ARTIFACT_IDENTITY_SHA256':'owner'}
  self.assertEqual(runtime_identity_environment(env),globals['runtime_environment'](env))
if __name__=='__main__':unittest.main()
