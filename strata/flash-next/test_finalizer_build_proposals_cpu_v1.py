import copy,json,tempfile,unittest
from pathlib import Path
import prepare_native43_build_proposal_v1 as n
import prepare_pti_finalizer_build_proposal_v1 as t
class Controls(unittest.TestCase):
 def test_exact_native43_roster_and_parent_rows(self):
  v=n.proposal();parent=json.loads(n.PARENT.read_bytes());self.assertEqual((len(v['overlay_files']),len(v['expected_patched_source_sha256']),len(v['added_header_payloads']),len(v['patches']),len(v['build_targets']),len(v['runtime_python_sources'])),(70,70,32,43,8,6));self.assertEqual(set(v['overlay_files']),set(v['expected_patched_source_sha256']));self.assertEqual(v['patches'][:-1],parent['patches']);self.assertEqual(v['added_header_payloads'][:-1],parent['added_header_payloads']);self.assertEqual(v['runtime_python_sources'],parent['runtime_python_sources']);self.assertFalse(v['source41_or42_runtime_proof_transferred'])
 def test_final_materialized_headers_distinct_from_historical_payloads(self):
  v=n.proposal();self.assertFalse(v['historical_payloads_are_final_materialized_header_proof']);self.assertEqual(v['expected_patched_source_sha256'][n.native.HEADER],n.sha(n.HERE/'pti_native_finalize43_v1.hpp'));self.assertEqual(v['expected_patched_source_sha256'][n.native.parent.H],n.sha(n.HERE/'host_critical_path42_header_v1.hpp'))
 def test_full_tool_closure_and_exact_three_modified_added_files(self):
  v=t.proposal();self.assertEqual(v['source_file_count'],895);self.assertEqual(v['original_tracked_count'],893);self.assertEqual(set(v['expected_patched_source_sha256']),{t.change.H,t.change.T,t.change.GUARD});self.assertEqual(v['expected_full_source_sha256'][t.change.GUARD],t.sha(t.HERE/'pti_model_finalize_guard_v1.hpp'));self.assertEqual(v['build_targets'],['unitrace','unitrace_tool']);self.assertFalse(v['old_budget_ELF_build_receipt_transferred']);self.assertFalse(v['actual_build_executed'])
 def test_private_recipe_actual_source_and_label_no_devices_or_weights(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);source=root/'new-source';source.mkdir();output=root/'new-build';old_source=t.original.SOURCE;old_label=t.original.MANIFEST_SHA;cmd=t.recipe(source,output,'a'*64);self.assertIn(str(source)+':/src:ro',cmd);self.assertIn('b70.pti.source='+'a'*64,cmd);self.assertNotIn('--device',cmd);self.assertNotIn('--gpus',cmd);self.assertFalse(any('/model:'in x or'/pack:'in x for x in cmd));self.assertEqual(t.original.SOURCE,old_source);self.assertEqual(t.original.MANIFEST_SHA,old_label)
if __name__=='__main__':unittest.main()
