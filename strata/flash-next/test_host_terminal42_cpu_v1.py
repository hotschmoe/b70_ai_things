import hashlib,re,tempfile,unittest
from pathlib import Path
import prepare_host_terminal42_v1 as p
class Controls(unittest.TestCase):
 def test_exact_patch_and_two_file_undo(self):
  old,new=p.changes();self.assertEqual(p.PATCH.read_text(),p.patch_text());self.assertEqual(new[p.G].replace(p.REPLACEMENT,p.ANCHOR),old[p.G]);self.assertEqual(new[p.H].replace(p.NEW,p.OLD),old[p.H]);self.assertEqual((p.HERE/'host_critical_path42_header_v1.hpp').read_text(),new[p.H])
 def test_normal_fast_exit_and_failure_exits_unchanged(self):
  old,new=p.changes();self.assertEqual(new[p.G].count('std::_Exit(0)'),old[p.G].count('std::_Exit(0)'));self.assertEqual(new[p.G].count('std::_Exit(1)'),old[p.G].count('std::_Exit(1)'));self.assertIn(p.REPLACEMENT,new[p.G])
 def test_no_new_GPU_wait_query_or_arithmetic_call(self):
  old,new=p.changes()
  for path in(p.G,p.H):
   calls=lambda text:re.findall(r'\b(?:wait|wait_and_throw|queues_wait_and_throw|ext_oneapi_graph|zeDeviceGetGlobalTimestamps|zeEventQueryKernelTimestamp)\s*\(',text)
   self.assertEqual(calls(old[path]),calls(new[path]))
 def test_header_disables_before_owned_delete_and_callback_remains(self):
  _,new=p.changes();header=new[p.H];self.assertIn('std::atexit(exit_export)',header);body=header.split('inline void exit_export()',1)[1].split('inline bool init',1)[0];self.assertLess(body.index('s.on.store(false'),body.index('delete[] s.events'));self.assertIn('if(!enabled()||!writer())return;',body)
 def test_source42_fresh_build_roster_no_old_runtime_transfer(self):
  v=p.proposal();self.assertEqual(len(v['patches']),42);self.assertEqual(len(v['expected_patched_source_sha256']),69);self.assertEqual(len(v['build_targets']),8);self.assertFalse(v['old_source41_runtime_proof_transferred']);self.assertFalse(v['PTI_destructor_flush_fixed']);self.assertEqual(v['expected_patched_source_sha256'][p.H],hashlib.sha256((p.HERE/'host_critical_path42_header_v1.hpp').read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
