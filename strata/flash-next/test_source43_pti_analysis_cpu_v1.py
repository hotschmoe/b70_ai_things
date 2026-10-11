"""Synthetic raw interval/coherence/coverage boundaries; no model or GPU reads."""
import copy,unittest
import analyze_source43_pti_latency_v1 as a
class Controls(unittest.TestCase):
 def row(self,i,start,end,device=7,mask=(1<<64)-1):return {'instance_id':i,'device_handle':device,'timer_frequency_hz':1000000000,'timer_mask':mask,'raw_kernel_start_ticks':start,'raw_kernel_end_ticks':end}
 def test_overlap_union_not_sum_and_nearest_rank_gaps(self):
  rows=[self.row(1,10,30),self.row(2,20,40),self.row(3,50,60),self.row(4,60,80),self.row(5,100,120)];r=a.raw_device_metrics(rows)[0];self.assertEqual(r['observed_busy_union_ticks'],80);self.assertEqual(r['raw_intervals_ticks'],[[10,40],[50,80],[100,120]]);self.assertEqual(r['observed_gaps_ticks'],[10,20]);self.assertEqual(r['gap_P50_ticks'],10);self.assertEqual(r['gap_P95_ticks'],20);self.assertFalse(r['physical_idle_qualified']);self.assertFalse(r['host_clock_alignment_used'])
 def test_device_domains_not_added_or_host_reordered(self):
  rows=[self.row(1,50,60),self.row(2,10,20),self.row(3,1,99,8)];r=a.raw_device_metrics(rows);self.assertEqual(len(r),2);self.assertEqual(r[0]['observed_busy_union_ticks'],20);self.assertEqual(r[0]['producer_record_start_order_reversals'],1);self.assertEqual(r[1]['observed_busy_union_ticks'],98)
 def test_wrap_or_ambiguous_epoch_has_no_busy_idle_result(self):
  for rows in([self.row(1,250,5,mask=255)],[self.row(1,1,10,mask=255),self.row(2,190,200,mask=255)]):
   r=a.raw_device_metrics(rows)[0];self.assertTrue(r['raw_linear_epoch_ambiguous']);self.assertIsNone(r['observed_busy_union_ticks']);self.assertIsNone(r['observed_gaps_ticks']);self.assertFalse(r['physical_idle_qualified'])
 def test_boolean_duplicate_instance_or_bad_mask_refused(self):
  for rows in([self.row(True,1,2)],[self.row(1,1,2),self.row(1,2,3)],[self.row(1,1,2,mask=17)]):
   with self.assertRaises(ValueError):a.raw_device_metrics(rows)
 def test_zero_duration_is_not_fabricated_busy(self):
  r=a.raw_device_metrics([self.row(1,20,20)])[0];self.assertEqual(r['observed_busy_union_ticks'],0);self.assertEqual(r['observed_gaps_ticks'],[]);self.assertIsNone(r['gap_P50_ticks'])
 def test_missing_successful_public_admission_refuses_before_raw(self):
  # The full source/producer wiring is separately source-closed. This checks
  # that an arbitrary false saved admission cannot turn raw observations into
  # qualified output, with only tiny metadata reads mocked.
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'report.json').write_text(json.dumps({'passed':True,'errors':[],'phase':'host-latency'}));public=root.parent/(root.name+'-admit.json');public.write_text('{"passed":false}')
   try:
    with patch.object(a,'source_binding',return_value='CPU_SOURCE'):
     with self.assertRaises(ValueError):a.analyze(root,public,a.host.bound(root/'report.json')['sha256'],a.host.bound(public)['sha256'])
   finally:public.unlink()
if __name__=='__main__':unittest.main()
