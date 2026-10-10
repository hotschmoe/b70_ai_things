"""Observed offline commands and bounded census source; no actual tool invocation."""
import ast,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import half37_runtime_ocloc_probe_recipe_v1 as r
class Controls(unittest.TestCase):
 def test_observed_top_help_allows_only_declared_help_version_no_target(self):
  self.assertEqual(r.probe_argv(),{name:[r.TOOL,*args]for name,args in {'compile-help':['compile','--help'],'disasm-help':['disasm','--help'],'ids-help':['ids','--help'],'query-help':['query','--help'],'version':['--version']}.items()})
  for row in r.probe_argv().values():self.assertNotIn('-device',row);self.assertNotIn('-file',row)
 def test_pinned_runtime_census_proposal_separate_from_compile(self):
  with patch.object(r,'inventory_binding',return_value={'actual':'observed'}),patch.object(r,'original_library_specs',return_value=r.REQUIRED_LIBRARIES):result=r.recipes('/tmp/declared-owned-output',123)
  self.assertEqual(len(result['commands']),6);self.assertIsNone(result['device_target_selected']);self.assertFalse(result['offline_compile_or_disassembly_requested']);self.assertFalse(result['original_runtime_JIT_ISA_observed']);self.assertEqual(result['expected_original_library_bytes'],r.REQUIRED_LIBRARIES)
  for row in result['commands'].values():self.assertIn('--entrypoint',row);self.assertIn('--network',row);self.assertIn('SYCL_CACHE_PERSISTENT=0',row);self.assertNotIn('/model',str(row));self.assertNotIn('/pack',str(row))
 def test_exact_root_identity_bool_refused_before_any_binding(self):
  with patch.object(r,'inventory_binding')as observed:
   self.assertRaises(ValueError,r.recipes,'/tmp/no-created',True);observed.assert_not_called()
 def test_single_mounted_census_entry_is_stdlib_only(self):
  tree=ast.parse(Path(r.__file__).read_text());top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
  self.assertEqual([a.name for n in top if isinstance(n,ast.Import)for a in n.names],['hashlib','json','shlex','stat','subprocess']);self.assertEqual([n.module for n in top if isinstance(n,ast.ImportFrom)],['pathlib'])
  source=Path(r.__file__).read_text();self.assertIn('before.st_size<=128<<20',source);self.assertIn("stream.read(1<<20)",source);self.assertIn('ldd_is_not_dynamic_IGC_dlopen_or_JIT_map_authority',source);self.assertNotIn('ctypes.CDLL',source)
if __name__=='__main__':unittest.main()
