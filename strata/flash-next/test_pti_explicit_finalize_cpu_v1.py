import re,unittest
from pathlib import Path
import prepare_pti_explicit_finalize_v1 as tool
import prepare_pti_native_hook43_v1 as native
class Controls(unittest.TestCase):
 def test_exact_tool_patch_fresh_only(self):
  old,new=tool.changes();self.assertEqual((tool.HERE/'pti-explicit-model-finalizer-v1.patch').read_text(),tool.patch_text());self.assertEqual(set(new),{tool.H,tool.T,tool.GUARD});self.assertIn('512u << 20',new[tool.H]);self.assertIn('4096',new[tool.H])
 def test_clock_close_has_terminal_bridge_lock_and_no_reopen(self):
  _,new=tool.changes();header=new[tool.H];self.assertIn('if (closed) return !close_error;',header);self.assertIn('EmitLocked("host_clock_bridge", {{"site",2}',header);self.assertIn('const int result=::close(fd);fd=-1;closed=true',header);self.assertIn('if (closed) Fail("metadata emission after explicit close")',header)
 def test_teardown_and_old_destructor_remain_owned_idempotent(self):
  old,new=tool.changes();self.assertIn('void DESTRUCTOR Fini(void) {\n  Teardown();\n}',new[tool.T]);self.assertIn('[] { Teardown(); }, [] { return b70_pti_clock_v1::CloseOnce(); }',new[tool.T]);self.assertIn('b70_pti_finalize_v1::state().Initialize()',new[tool.T]);self.assertIn('owner_pid',new[tool.GUARD]);self.assertIn('failed.store(true',new[tool.GUARD])
 def test_native_exact_byte_undo_and_default_off_no_resolution(self):
  old,new=native.changes();self.assertEqual(native.PATCH.read_text(),native.patch_text());header=new[native.HEADER];self.assertLess(header.index('if(!flag||'),header.index('::dlsym'));self.assertIn('RTLD_DEFAULT,"b70_pti_model_finalize_v1"',header);self.assertIn('/pti/lib/libunitrace_tool.so',header);self.assertIn('std::_Exit(0)',new[native.G]);self.assertIn('std::_Exit(1)',new[native.G])
 def test_no_new_native_gpu_api_or_arithmetic(self):
  old,new=native.changes();pattern=r'\b(?:wait|wait_and_throw|zeDeviceGetGlobalTimestamps|zeEventQueryKernelTimestamp)\s*\('
  self.assertEqual(re.findall(pattern,old[native.G]),re.findall(pattern,new[native.G]));self.assertEqual(old[native.G].count('std::exit('),new[native.G].count('std::exit('));self.assertNotIn('std::atexit(',new[native.HEADER])
 def test_guard_fixture_retains_partial_failure_not_false_success(self):
  fixture=(tool.HERE/'pti_model_finalize_cpu_fixture_v1.cpp').read_text();self.assertIn('Finalize(1,pid+1,tid',fixture);self.assertIn('!=5||tries!=2',fixture);self.assertIn('!=5||clocks!=2',fixture)
if __name__=='__main__':unittest.main()
