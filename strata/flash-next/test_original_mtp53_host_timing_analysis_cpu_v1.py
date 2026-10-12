import unittest
import analyze_original_mtp53_host_timing_v1 as a
class Controls(unittest.TestCase):
 def test_nested_exclusive_and_committed_token_join(self):
  rows=[]
  for T,pos in((2,2048),(2,2050)):
   for d,b,e in((0,100,500),(1,200,400)):
    rows.append({'scope':'decode','phase':'target_stage_run','device':str(d),'T':str(T),'position':str(pos),'tid':'1','begin_ns':str(b),'end_ns':str(e)})
  actual={'host_timing':{'rows':rows},'ids':[7,8,9,10],'windows':[{'remaining':'511','position':'2050','committed':'2','accepted':'1','input':'7'},{'remaining':'509','position':'2051','committed':'1','accepted':'0','input':'9'}]}
  v=a.summarize(actual);self.assertEqual(v['decode_windows'][0]['output_ids'],[8,9]);self.assertEqual(v['decode_windows'][1]['output_ids'],[10]);self.assertAlmostEqual(v['decode_windows'][0]['stage0_exclusive_host_ms'],.0002)
  self.assertEqual(v['decode_window_statistics']['accepted']['committed_tokens'],2);self.assertFalse(v['nested_totals_additive']);self.assertIn('decode/target_stage_run/device0',v['scope_phase_statistics']);self.assertNotIn('decode/target_stage_run',v['scope_phase_statistics'])
 def test_nearest_rank(self):self.assertEqual(a.stats([4,1,2,3])['p50_ms'],2);self.assertEqual(a.stats([4,1,2,3])['p95_ms'],4)
if __name__=='__main__':unittest.main()
