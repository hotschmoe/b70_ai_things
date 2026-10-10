import copy,json,struct,tempfile,unittest
from pathlib import Path
import collect_half37_native_bundle_v1 as c

class Controls(unittest.TestCase):
 def test_mapped_library_roster_paths_and_unknown_deleted(self):
  from half37_native_bundle_library_census_v1 import mapped_paths
  raw='1-2 r-xp 0 00:1 1 /helper/half37-native-bundle\n2-3 r-xp 0 00:1 2 /usr/lib/libx.so.1\n';self.assertEqual(len(mapped_paths(raw)),2);self.assertRaises(ValueError,mapped_paths,raw+'3-4 r-xp 0 00:1 3 /usr/lib/libgone.so (deleted)\n');self.assertRaises(ValueError,mapped_paths,raw.replace('/helper/half37-native-bundle','/foreign/helper'))
 def test_CPU_census_recipe_no_devices_or_models(self):
  from half37_native_bundle_library_census_v1 import recipe
  argv=recipe(Path('/runtime'),Path('/compile'));self.assertNotIn('--device',argv);self.assertNotIn('/models',str(argv));self.assertIn('--network',argv);self.assertIn('/opt/b70-c1-python/bin/python',argv)
 def test_library_file_size_guard_and_ELF_metadata(self):
  from half37_native_bundle_library_census_v1 import file_binding
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'elf';p.write_bytes(b'\x7fELF');self.assertEqual(file_binding(p,4)['ELF_magic'],'7f454c46');self.assertRaises(ValueError,file_binding,p,3)

 def fixture(self,root):
  names=sorted(c.NAMES);lines=[]
  for i,name in enumerate(names):lines.append(f'HALF37_BUNDLE_KERNEL name={name} handle=0x{i+10:x} module_uuid='+32*'0'+' kernel_uuid='+32*'0'+' executable_bundle_membership=1 actual_direct_launch_handle_observed=0')
  for i,name in enumerate(names):
   raw=bytearray(64);raw[:7]=b'\x7fELF\x02\x01\x01';struct.pack_into('<H',raw,18,205);(root/f'module-{i}.bin').write_bytes(raw);lines += [f'HALF37_BUNDLE_MODULE index={i} handle=0x{i+20:x} bytes=64 names=1',f'HALF37_BUNDLE_NAME module={i} name={name}']
  lines.append('HALF37_BUNDLE_RESULT modules=3 bytes=192 new_ELF_only=1 historical_JIT_observed=0 actual_direct_launch_module_proven=0')
  lines += [f'HALF37_FRAME route={i} graph_replay={i} fields=4 own_input_restored=1 values=512'for i in range(3)];lines.append('HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0');return '\n'.join(lines)+'\n'
 def test_exact_bundle_scope_uuid_zero_does_not_claim_launch(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);p=c.parse(self.fixture(root),root);self.assertEqual(len(p['modules']),3);self.assertFalse(p['actual_direct_launch_module_association_proven']);self.assertFalse(p['historical_JIT_code_observed'])
 def test_missing_duplicate_or_wrong_kernel_role_refused(self):
  for mode in ('missing','duplicate','wrong'):
   with tempfile.TemporaryDirectory()as temp:
    root=Path(temp);log=self.fixture(root)
    if mode=='missing':log='\n'.join(x for x in log.splitlines()if not x.startswith('HALF37_BUNDLE_KERNEL name=_ZTS10Half37Load'))+'\n'
    elif mode=='duplicate':log+=next(x for x in log.splitlines()if x.startswith('HALF37_BUNDLE_KERNEL'))+'\n'
    else:log=log.replace('_ZTS10Half37Load','FOREIGN')
    with self.assertRaises(ValueError):c.parse(log,root)
 def test_extra_file_truncation_foreignELF_or_wrong_bytes_refused(self):
  for mode in ('extra','truncated','machine','bytes'):
   with tempfile.TemporaryDirectory()as temp:
    root=Path(temp);log=self.fixture(root)
    if mode=='extra':(root/'extra').write_bytes(b'x')
    elif mode=='truncated':(root/'module-0.bin').write_bytes(b'\x7fELF')
    elif mode=='machine':raw=bytearray((root/'module-0.bin').read_bytes());struct.pack_into('<H',raw,18,62);(root/'module-0.bin').write_bytes(raw)
    else:log=log.replace('bytes=64','bytes=65',1)
    with self.assertRaises(ValueError):c.parse(log,root)
 def test_symlink_native_artifact_refused(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);log=self.fixture(root);raw=(root/'module-0.bin').read_bytes();(root/'module-0.bin').unlink();(root/'original').write_bytes(raw);(root/'module-0.bin').symlink_to(root/'original');self.assertRaises(ValueError,c.parse,log,root)
 def test_normal_typed_terminal_and_errors(self):
  state={'Status':'exited','Error':'','Running':False,'Paused':False,'Restarting':False,'OOMKilled':False,'Dead':False,'ExitCode':0,'Pid':0};c.terminal(state)
  for key,value in [('ExitCode',False),('Error','contradictory'),('Running',True),('Pid',False),('Status','dead')]:
   x={**state,key:value};self.assertRaises(ValueError,c.terminal,x)
 def test_actual_kernel_math_lines_preserved(self):
  src=(c.HERE/'half37_native_bundle_diagnostic_v1.cpp').read_text();old=(c.HERE/'owned_indexer_half_discrimination_gpu_v2.cpp').read_text();self.assertEqual([x.strip()for x in src.splitlines()if 'q.parallel_for<Half37'in x],[x.strip()for x in old.splitlines()if 'q.parallel_for<Half37'in x]);self.assertEqual(c.sha(c.HERE/'half37_native_bundle_diagnostic_v1.cpp'),c.SOURCE_SHA)
 def test_duplicate_json_and_bounded_pre_read(self):
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'x';p.write_text('{"key":1,"key":2}');self.assertRaises(ValueError,c.read,p);p.write_bytes(b'123');self.assertRaises(ValueError,c.data,p,2)
 def test_actual_runtime_recipe_has_no_model_or_interposer(self):
  recipe=c.runtime_recipe('b70-half37-jit-1',Path('/out'),Path('/compile'));self.assertIn('SYCL_CACHE_PERSISTENT=0',recipe);self.assertNotIn('LD_PRELOAD',str(recipe));self.assertNotIn('/models',str(recipe));self.assertRaises(ValueError,c.runtime_recipe,'foreign',Path('/out'),Path('/compile'))

if __name__=='__main__':unittest.main()
