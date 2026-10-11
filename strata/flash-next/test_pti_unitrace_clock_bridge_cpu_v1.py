import copy,json,re,time,unittest
from pathlib import Path
import prepare_pti_unitrace_clock_bridge_v1 as p

class BridgeControls(unittest.TestCase):
 def test_exact_frozen_source_patch(self):
  self.assertEqual((p.ROOT/'strata/flash-next/pti-unitrace-clock-bridge-v1.patch').read_text(),p.patch_text());self.assertEqual(set(p.changes()),{p.HEADER,p.ZE,p.TIMER})
 def test_all_GPU_queries_and_waits_identical(self):
  for name,new in p.changes().items():
   old=(p.BASE/name).read_text()
   self.assertEqual(re.findall(r'ZE_FUNC\([^)]*\)\([^;]*;',new),re.findall(r'ZE_FUNC\([^)]*\)\([^;]*;',old))
  self.assertEqual(p.changes()[p.ZE].count('BeginQueryBracket();'),6)
 def test_six_exact_GPU_query_enclosures_and_no_math_edits(self):
  old=(p.BASE/p.ZE).read_text();new=p.changes()[p.ZE]
  for site in range(1,7):
   self.assertEqual(new.count('const auto b70_bridge_query_'+str(site)+' ='),1)
   self.assertEqual(new.count('EndQueryBracket(b70_bridge_query_'+str(site)+','),1)
  stripped=re.sub(r'(?m)^[ \t]*const auto b70_bridge_query_\d+ = b70_pti_clock_v1::BeginQueryBracket\(\);\n','',new)
  stripped=re.sub(r'(?m)^[ \t]*b70_pti_clock_v1::EndQueryBracket\([^\n]+\);\n','',stripped)
  self.assertEqual(stripped,old)
 def test_origin_original_CPU_queries_unchanged(self):
  old=(p.BASE/p.TIMER).read_text();new=p.changes()[p.TIMER]
  self.assertEqual(new.replace('\n        b70_pti_clock_v1::HostBridge(1);',''),old)
 def test_default_OFF_sampling_and_invalid_flag_gate(self):
  h=p.changes()[p.HEADER]
  self.assertIn('flag == nullptr || flag[0] ==',h);self.assertIn('std::strcmp(flag, "0") == 0) return false;',h)
  self.assertIn('std::strcmp(flag, "1") != 0 || !Enabled()',h)
  before=h[h.index('inline QueryBracket BeginQueryBracket()'):h.index('inline void EndQueryBracket(')]
  self.assertLess(before.index('if (!BridgeEnabled())'),before.index('ReadHostClock(CLOCK_MONOTONIC_RAW)'))
  finish=h[h.index('inline void EndQueryBracket('):];self.assertLess(finish.index('if (!bracket.enabled) return;'),finish.index('ReadHostClock(CLOCK_MONOTONIC_RAW)'))
 def test_three_clock_uncertainty_bracket(self):
  value=p.interval({'raw_before_ns':100,'monotonic_ns':150,'raw_after_ns':110});self.assertEqual(value['monotonic_minus_raw_offset_interval_ns'],[40,50]);self.assertEqual(value['bracket_width_ns'],10);self.assertFalse(value['interpolation_or_drift_bound_assumed'])
 def test_negative_offset_zero_width_and_long_delay_retained(self):
  self.assertEqual(p.interval({'raw_before_ns':200,'monotonic_ns':150,'raw_after_ns':210})['monotonic_minus_raw_offset_interval_ns'],[-60,-50])
  self.assertEqual(p.interval({'raw_before_ns':100,'monotonic_ns':150,'raw_after_ns':100})['bracket_width_ns'],0)
  self.assertEqual(p.interval({'raw_before_ns':100,'monotonic_ns':150,'raw_after_ns':100000100})['bracket_width_ns'],100000000)
 def test_actual_tiny_host_CPU_bridge(self):
  # Only this Python test process's CPU clocks, not a native/model/GPU observation.
  a=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);m=time.clock_gettime_ns(time.CLOCK_MONOTONIC);b=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
  result=p.interval({'raw_before_ns':a,'monotonic_ns':m,'raw_after_ns':b});self.assertGreaterEqual(result['bracket_width_ns'],0)
 def test_reordered_bool_foreign_extra_fields_rejected(self):
  original={'raw_before_ns':100,'monotonic_ns':150,'raw_after_ns':110}
  for bad in [{**original,'raw_after_ns':99},{**original,'raw_before_ns':True},{**original,'monotonic_ns':-1},{**original,'foreign':0}]:
   with self.assertRaises(ValueError):p.interval(bad)
 def test_failed_GPU_result_never_reads_uninitialized_pair(self):
  z=p.changes()[p.ZE]
  self.assertEqual(z.count('status == ZE_RESULT_SUCCESS ?'),12)
  h=p.changes()[p.HEADER];self.assertIn('uint64_t(status == 0)',h);self.assertIn('"return_code", status',h)
 def test_close_sample_before_original_close_and_error_path(self):
  h=p.changes()[p.HEADER];destruct=h[h.index('  ~Writer()'):h.index('  void Emit(')]
  self.assertLess(destruct.index('ReadHostBridge()'),destruct.index('::close(fd)'))
  self.assertIn('"final close failed"',destruct);self.assertIn('if (fd >= 0)',destruct)
if __name__=='__main__':unittest.main()
