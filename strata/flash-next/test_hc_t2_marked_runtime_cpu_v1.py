import unittest
import hc_t2_route_runtime_v1 as r
class Controls(unittest.TestCase):
 def test_actual_capture_plus_submission_counts(self):
  actual={'windows':[{'remaining':'511','position':'2050','committed':'2'},{'remaining':'509','position':'2051','committed':'1'}]}
  text=''
  for d in(0,1):
   common='pid=1 owner=0x%d graph=0x%d device=%d lb=%d le=%d enabled=1 eligible=48 selected=48 graph_kind=SYCL_host_identity'%(d+1,d+10,d,d*24,(d+1)*24)
   text+='HC_T2_ROUTE_CAPTURE '+common+'\n';text+='HC_T2_ROUTE_CLOSE '+common+' submitted=2 completed=2 first_pos=2048 last_pos=2050\n'
  text+='HC_T2_ROUTE_FINAL pid=1 failed=0 GPU_queries_added=0 GPU_waits_added=0\n'
  self.assertEqual(r.counts(text,actual,'on',1)['actual_T2_windows_per_stage'],{0:2,1:2})
  for bad in(text.replace('completed=2','completed=1',1),text.replace('eligible=48','eligible=0'),text.replace('selected=48','selected=0'),text.replace('submitted=2','submitted=0',1)):
   with self.assertRaises(ValueError):r.counts(bad,actual,'on',1)
if __name__=='__main__':unittest.main()
