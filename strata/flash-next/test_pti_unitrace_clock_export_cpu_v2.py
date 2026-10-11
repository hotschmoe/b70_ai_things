import unittest
import prepare_pti_unitrace_clock_export_v1 as old
import prepare_pti_unitrace_clock_export_v2 as new
from test_pti_unitrace_clock_export_cpu_v1 import ExportTests
class U64Tests(ExportTests):
 def test_patch_runtime_unchanged(self):self.assertEqual(new.patch_text(),old.patch_text())
 def test_actual_u64_equality_boundary(self):
  r=self.row(start=10,end=12,submit=10);r['timer_mask']=(1<<64)-1
  self.assertEqual(new.normalize(r),(1000,1008));self.assertNotEqual(old.normalize(r),new.normalize(r))
 def test_original_host_addition_wrap(self):
  r=self.row();r['submit_host_ns']=(1<<64)-4
  self.assertEqual(new.normalize(r),(4,24))
 def test_unrepresentable_original_cast_refuses(self):
  r=self.row(start=12,end=17,submit=13);r['timer_mask']=(1<<64)-1
  with self.assertRaises(ValueError):new.normalize(r)
if __name__=='__main__':unittest.main()
