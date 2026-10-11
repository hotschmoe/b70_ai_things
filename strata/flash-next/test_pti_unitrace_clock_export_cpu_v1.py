import re
import struct
import unittest
from pathlib import Path
import prepare_pti_unitrace_clock_export_v1 as p

class ExportTests(unittest.TestCase):
 def test_exact_patch_and_only_output_changes(self):
  generated=p.changes();self.assertEqual(len(generated),3)
  self.assertEqual((p.ROOT/'strata/flash-next/pti-unitrace-clock-export-v1.patch').read_text(),p.patch_text())
  for name in (p.ZE,p.TIMER):
   original=(p.BASE/name).read_text();s=generated[name]
   s=re.sub(r'\n\s*if \(b70_pti_clock_v1::Enabled\(\)\) b70_pti_clock_v1::Emit\([^;]+;','',s)
   s=re.sub(r'\n#include "(?:\.\./)?pti_unitrace_clock_export_v1.h"','',s)
   self.assertEqual(s,original)
 def test_no_added_GPU_queries_or_waits(self):
  generated=p.changes()
  for name in (p.ZE,p.TIMER):
   old=(p.BASE/name).read_text();new=generated[name]
   self.assertEqual(re.findall(r'ZE_FUNC\([^)]*\)',old),re.findall(r'ZE_FUNC\([^)]*\)',new))
   self.assertEqual(re.findall(r'\bclock_gettime\([^;]*;',old),re.findall(r'\bclock_gettime\([^;]*;',new))
 def test_required_values_and_owner_fields(self):
  z=p.changes()[p.ZE]
  for key in ('instance_id','submit_device_ticks','submit_host_ns','raw_kernel_start_ticks','raw_kernel_end_ticks','timer_mask','timer_frequency_hz','kernel_valid_bits','global_valid_bits','command_list_handle','queue_handle','device_handle','epoch_origin_ns','normalized_start_host_ns','normalized_end_host_ns'):
   self.assertIn('"'+key+'"',z)
  self.assertIn('command->instance_id_',z);self.assertIn('props.kernelTimestampValidBits',z)
 def test_default_OFF_before_allocation_lock_and_open(self):
  h=p.HEADER.read_text();method=h[h.index('  void Emit('):h.index('inline void Emit(')]
  self.assertLess(method.index('if (!Enabled()) return;'),method.index('std::lock_guard'))
  wrapper=h[h.index('inline void Emit('):];self.assertLess(wrapper.index('if (!Enabled()) return;'),wrapper.index('static Writer'))
  for value in ('O_EXCL','O_NOFOLLOW','O_CLOEXEC','0600','owner != pid','8u << 20','_exit(86)','exact declared per-process output template'):
   self.assertIn(value,h)
 def row(self,start=12,end=17,submit=10):
  return {'timer_mask':255,'timer_frequency_hz':250000000,'ns_per_cycle_ieee754_u64':struct.unpack('=Q',struct.pack('=d',4.0))[0],'raw_kernel_start_ticks':start,'raw_kernel_end_ticks':end,'submit_device_ticks':submit,'submit_host_ns':1000}
 def test_original_conversion_and_wrap_exact(self):
  self.assertEqual(p.normalize(self.row()),(1008,1028))
  self.assertEqual(p.normalize(self.row(start=2,end=7,submit=250)),(1032,1052))
  self.assertEqual(p.normalize(self.row(start=250,end=2,submit=249)),(1004,1036))
 def test_original_equal_submit_semantics_preserved(self):
  # Original code's strict > takes its overflow branch on equality.
  self.assertEqual(p.normalize(self.row(start=10,end=12,submit=10)),(2024,2032))
 def test_boolean_and_stale_scalar_rejected(self):
  r=self.row();r['submit_host_ns']=True
  with self.assertRaises(ValueError):p.normalize(r)
  r=self.row();r['timer_frequency_hz']=500000000
  with self.assertRaises(ValueError):p.normalize(r)
if __name__=='__main__':unittest.main()
