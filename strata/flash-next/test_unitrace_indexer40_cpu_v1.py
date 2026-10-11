import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import probe_unitrace_indexer40_v1 as p

class ProbeTests(unittest.TestCase):
 def test_actual_source_admission(self):self.assertEqual(len(p.source_binding()),64)
 def test_exact_current_supported_command(self):
  c=p.tool_argv();self.assertEqual(c[0],'/tool/bin/unitrace');self.assertEqual(c[-len(p.helper_argv()):],p.helper_argv())
  self.assertIn('--chrome-kernel-logging',c);self.assertIn('--chrome-call-logging',c);self.assertIn('--chrome-device-logging',c)
  self.assertNotIn('--output-dir-path',c);self.assertNotIn('--result-dir',c);self.assertNotIn('--metric-query',c)
  self.assertEqual(c[c.index('--output')+1],'/results/unitrace/trace')
 def test_only_new_install_adds_mount(self):
  with patch.object(Path,'stat')as stat:
   stat.return_value.st_gid=991;c=p.recipe('on',Path('/tmp/new-root'),'owner','a'*64)
  binds=[c[i+1]for i,x in enumerate(c)if x=='-v'];self.assertEqual(len(binds),5);self.assertEqual(len(set(binds)),5)
  self.assertIn(str(p.INSTALL)+':/tool:ro',binds);self.assertEqual([x for x in binds if x.endswith(':rw')],['/tmp/new-root:/results:rw'])
  self.assertFalse(any('models' in x or 'source-pack'in x for x in binds));self.assertIn('ZE_AFFINITY_MASK=0',c)
 def test_helper_args_unchanged(self):
  self.assertEqual(p.helper_argv(),['/leaf/indexer-materialized40','--inputs','/inputs','--output','/results/build/native-output','--maps-prefix','/results/build/idxmat40'])
 def test_missing_hard_library_rejected(self):
  r={'ldd_command':['/usr/bin/ldd',p.TOOL],'ldd_return_code':0,'ldd_stdout':'liba.so => not found','files':[{}],'explicit_auxiliary_scope':None,'actual_dynamic_metadata':[]}
  with self.assertRaisesRegex(ValueError,'hard dependency'):p.dependency_record_binding(p.TOOL,r)
 def test_no_vtune_optional_exception(self):
  r={'ldd_command':['/usr/bin/ldd',p.TOOL_LIBRARY],'ldd_return_code':0,'ldd_stdout':'okay','files':[{}],'explicit_auxiliary_scope':{'passed':True},'actual_dynamic_metadata':[]}
  with self.assertRaises(ValueError):p.dependency_record_binding(p.TOOL_LIBRARY,r)
 def test_aslr_only_projection_and_injection_identity(self):
  row={'ldd_stdout':'lib.so (0x123)','files':[{'sha256':'a'}]};a={k:copy.deepcopy(row)for k in('helper_libraries','tool_libraries','injection_libraries')};b=copy.deepcopy(a)
  for v in b.values():v['ldd_stdout']='lib.so (0x456)'
  p.runtime_equal(a,b);b['injection_libraries']['files'][0]['sha256']='b'
  with self.assertRaises(ValueError):p.runtime_equal(a,b)
 def test_strict_terminal_state(self):
  a={'RestartCount':0,'State':{'ExitCode':0,'Pid':0,'Status':'exited','Error':'','Running':False,'OOMKilled':False,'Dead':False,'Paused':False,'Restarting':False}}
  p.normal_terminal(a)
  for field,value in [('ExitCode',False),('Pid',False),('Error','failed'),('Dead',True)]:
   b=copy.deepcopy(a);b['State'][field]=value
   with self.assertRaises(ValueError):p.normal_terminal(b)
 def test_evidence_regular_and_stable_own_read(self):
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'a';f.write_bytes(b'');self.assertEqual(p.file_binding(f)['bytes'],0);alias=Path(d)/'alias';alias.symlink_to(f)
   with self.assertRaises(ValueError):p.file_binding(alias)

if __name__=='__main__':unittest.main()
