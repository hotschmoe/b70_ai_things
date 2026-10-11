"""Actual-shaped Docker reorder and exact identity environment controls."""
import copy,unittest
from full_cache_shared_memory_capture_v7 import recipe_binding
from run_full_cache_shared_runtime_v7 import runtime_identity_environment
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
  from full_cache_shared_actor_retirement_v7 import external_recovery
  obj,cmd=self.fixture();obj.update(Name='/owned',Id='container-id');obj['State']={'Running':False,'ExitCode':2,'OOMKilled':False,'Error':''};obj['Config']['Labels']={'owner':'original'};cmd[2:2]=['--label','owner=original']
  owner={'pid':123,'start_ticks':456,'session':123,'state':'S'}
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary)
   for name,value in [('launch.command.json',cmd),('actor-owner.json',owner),('external-owned-terminal-removal.json',{'schema':1,'actor_owner':owner,'command':cmd,'name':'owned','normal_qualification':False,'terminal_inspection':obj,'container_id':'container-id','removal_receipt':{'command':['docker','rm','container-id'],'return_code':0,'error':None}})]:
    (root/name).write_text(json.dumps(value))
   with patch('full_cache_shared_runtime_v7.c1.leased'):
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
 def test_every_signed_frontend_family_uses_projection(self):
  import ast
  from pathlib import Path
  here=Path(__file__).parent
  for name in ('run_full_cache_shared_runtime_v7.py','run_full_cache_shared_batch0_persisted_v7.py'):
   tree=ast.parse((here/name).read_bytes());updates=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='update'and isinstance(n.func.value,ast.Subscript)and isinstance(n.func.value.slice,ast.Constant)and n.func.value.slice.value=='runtime']
   self.assertEqual(len(updates),1);env=next(k.value for k in updates[0].keywords if k.arg=='env');self.assertIsInstance(env,ast.Call);self.assertTrue((isinstance(env.func,ast.Name)and env.func.id=='runtime_identity_environment')or(isinstance(env.func,ast.Attribute)and env.func.attr=='runtime_identity_environment'))
 def test_actual_pinned_validate_runtime_actor_and_persisted_cpu_fixtures(self):
  import importlib.util,tempfile,hashlib,copy
  from pathlib import Path
  path=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo/source/serve/artifact_identity.py');self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'47619fcd5def60d13861773b359e9bab35499212f858ba649ce6462feb15027f');spec=importlib.util.spec_from_file_location('actual_pinned_artifact_identity_cpu_fixture',path);actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
  # Host test Python lacks image-only regex metadata. Supply explicitly
  # synthetic package-version lookup; validate_runtime and its source/byte
  # predicates remain the actual unchanged consumed production function.
  actual.package_version=lambda name:'CPU-fixture-version:'+name
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);exe=root/'synthetic-exe';exe.write_bytes(b'CPU fixture only, never an ELF or model proof')
   for name in actual.FILES:(root/name).write_bytes(b'CPU fixture '+name.encode('ascii'))
   for batch in (2,0):
    args=['--batch',str(batch),'--prompt-cache','3'];env={'STRATA_BATCH_FIDELITY_DIAG':'1','STRATA_FULL_CACHE_OBSERVER38':'1','SYCL_UR_TRACE':'2','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','ZE_AFFINITY_MASK':'0.0,1.0','PATH':'/synthetic','STRATA_ARTIFACT_IDENTITY_SHA256':'signed'};cfg={'args':args,'env':env};saved=copy.deepcopy(cfg)
    runtime={'exe_sha256':actual.sha256_file(exe),'args':args,'env':runtime_identity_environment(env),'python_versions':actual.python_contract(),'python_sources':actual.source_contract()};identity={'manifest':{'runtime':runtime,'tokenizer_files':{name:actual.sha256_file(root/name)for name in actual.FILES}},'tokenizer_path':str(root)}
    self.assertIsNone(actual.validate_runtime(identity,str(exe),cfg['args'],cfg['env']));self.assertEqual(cfg,saved)
    bad=copy.deepcopy(identity);bad['manifest']['runtime']['env']=dict(env);self.assertRaisesRegex(ValueError,'engine environment differs',actual.validate_runtime,bad,str(exe),args,env)
    changed=dict(env,STRATA_BATCH_FIDELITY_DIAG='0');self.assertRaisesRegex(ValueError,'engine environment differs',actual.validate_runtime,identity,str(exe),args,changed)
if __name__=='__main__':unittest.main()
