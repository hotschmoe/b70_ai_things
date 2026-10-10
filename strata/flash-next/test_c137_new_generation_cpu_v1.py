"""Source-only new generation and diagnostic-OFF controls; no model/device IO."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import c1_serve_controller_combined_v137 as c
import qualify_c1_serving_combined_v137 as q
import prepare_batch_stage_stamp37_v1 as p
import prepare_host_critical_path36_v1 as h
import test_c1_serve_controller_combined_v137 as controller_cpu
import test_qualify_c1_serving_combined_v137 as parent_cpu
class MarkerControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.engine=Path(self.tmp.name)
  _,self.rows=h.reconstruct();self.rows[p.V]=self.rows[p.V].replace(p.ANCHOR,p.ANCHOR.replace('        return true;',p.HOOK+'        return true;'))
  for name,text in self.rows.items():
   path=self.engine/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 def test_actual_reconstructed_source36_and37_presence(self):
  r=c.compiled_source_markers(self.engine);self.assertTrue(r['bounded_host_critical_path_trace36']);self.assertTrue(r['earlier_stage_batch_observer_stamp37']);self.assertFalse(r['actual_trace36_observed']);self.assertFalse(r['actual_paired_observer37_qualified'])
 def test_source35_and_source36_missing_stamp37_refused(self):
  path=self.engine/'source'/p.V
  for text in [(p.SDK/p.V).read_text(),h.reconstruct()[1][p.V]]:
   path.write_text(text)
   with self.assertRaises(ValueError):c.compiled_source_markers(self.engine)
 def test_stamp_before_handoff_and_missing_head_refused(self):
  path=self.engine/'source'/p.V;original=self.rows[p.V]
  path.write_text(original.replace(p.HOOK,'',1).replace('        float* hout =',p.HOOK+'        float* hout =',1))
  with self.assertRaises(ValueError):c.compiled_source_markers(self.engine)
  path.write_text(original[:original.index('    // ---- the head, T columns')]+original[original.index('    // ---- the head, T columns'):].replace(p.HOOK.strip(),'/* missing head stamp */',1))
  with self.assertRaises(ValueError):c.compiled_source_markers(self.engine)
 def test_hosttrace_header_missing_flag_refused(self):
  path=self.engine/'source'/h.HEADER;path.write_text(path.read_text().replace('STRATA_CRITICAL_PATH_TRACE','CPU_CHANGED_FLAG'))
  with self.assertRaises(ValueError):c.compiled_source_markers(self.engine)
 def test_h36_pair_and_baseline_off_eager_absent(self):
  for flag in ('STRATA_CRITICAL_PATH_TRACE','STRATA_BATCH_FIDELITY_DIAG'):
   with self.assertRaises(ValueError):c.baseline_profile_gate(['--batch','0'],{flag:'1'})
  for value in ('0','1'):
   with self.assertRaises(ValueError):c.baseline_profile_gate(['--batch','0'],{'STRATA_VERIFY_EAGER':value})
class FreshControls(unittest.TestCase):
 def setUp(self):
  self.fixture=controller_cpu.GenerationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.receipt=self.fixture.receipt
 def gate(self):return self.fixture.gate()

 def test_source35_actual_old_plan_never_upload_or_page(self):
  self.receipt['plan_sha256']=p.BASE_SHA
  with patch.object(c,'upload_gate',side_effect=AssertionError('oldSDK mustnot access upload')),patch.object(c,'original_page_sentinel',side_effect=AssertionError('oldSDK mustnot read pages')):
   with self.assertRaises(ValueError):self.gate()
 def test_missing_trace36_header_refused(self):
  self.receipt['patched_source_sha256'].pop(h.HEADER)
  with self.assertRaises(ValueError):self.gate()
 def test_source37_standalone_plan_refused_combined_adapter(self):
  self.receipt['plan_sha256']='2cb8c2559970cd3830e58081df97b213b6d4e07599ad98cdad6c76ab07c2f972'
  with self.assertRaises(ValueError):self.gate()
class NewFinalControls(unittest.TestCase):
 def setUp(self):
  self.fixture=parent_cpu.FinalTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.prepared=self.fixture.prepared;self.final=self.fixture.final
 def validate(self):return self.fixture.validate()

 def test_old_C113_final_proof_refused(self):
  self.final['c1_parent_generation']=13
  with self.assertRaises(AssertionError):self.validate()
 def test_each_new_source_marker_required(self):
  for flag in ('bounded_host_critical_path_trace36','earlier_stage_batch_observer_stamp37'):
   self.prepared['combined_generation'][flag]=False
   with self.assertRaises(AssertionError):self.validate()
   self.prepared['combined_generation'][flag]=True
if __name__=='__main__':unittest.main()
