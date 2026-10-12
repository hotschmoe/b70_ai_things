import unittest
import analyze_source52_selected_kernel_unions_v1 as a
class Controls(unittest.TestCase):
 def test_overlap_union_and_device_domains(self):
  rows={1:[1,0,10,0,0,'HcProjectionKernel'],2:[1,5,15,0,0,'gemm_f16'],3:[2,100,120,0,0,'gdn_step']};v=a.detail(rows,{1:{'timer_frequency_hz':1000,'descriptor_id':0},2:{'timer_frequency_hz':2000,'descriptor_id':1}});self.assertEqual(v[0]['all_observed_operations']['observed_union_ms'],15);self.assertEqual(v[1]['all_observed_operations']['observed_union_ms'],10);self.assertFalse(v[0]['physical_idle_qualified'])
 def test_named_families_and_no_temporal_split(self):
  self.assertEqual(a.family('iq_dequant_f16'),'dequantization');self.assertEqual(a.family('strata::kernels::HcProjectionKernel<true>'),'HC_native');self.assertEqual(a.family('gemm_bf16'),'GEMM_vendor_or_named');self.assertEqual(a.family('arbitrary_vendor_kernel'),'other_named_operations')
if __name__=='__main__':unittest.main()
