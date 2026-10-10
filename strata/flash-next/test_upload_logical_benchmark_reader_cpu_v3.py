"""Finite ordered edges and exact typed timed call controls; no runtime payload."""
import copy,unittest
import validate_upload_logical_benchmark_v3 as r
class Controls(unittest.TestCase):
 def test_full_byte_shape_change_and_reversed_edge_are_refused(self):
  expected=[{'path':'/synthetic','sha256':'0'*64,'stat_before':[1,2,3,4,5]}];rows=[{'label':label,'started_epoch':i+1,'finished_epoch':i+1.5,'rows':expected}for i,label in enumerate(['entry','predevice','postoperation'])];r.byte_edges(rows,['entry','predevice','postoperation'],expected,0,5)
  for mutate in ('bytes','time','bool','duplicate'):
   bad=copy.deepcopy(rows)
   if mutate=='bytes':bad[1]['rows'][0]['sha256']='1'*64
   elif mutate=='time':bad[1]['started_epoch']=0
   elif mutate=='bool':bad[0]['finished_epoch']=True
   else:bad[1]['label']='entry'
   self.assertRaises(ValueError,r.byte_edges,bad,['entry','predevice','postoperation'],expected,0,5)
 def test_timing_flags_indices_and_nonfinite_are_not_inferred(self):
  row={'arm':'B','started_epoch':1,'setup_finished_epoch':2,'byte_seal_started_epoch':3,'finished_epoch':4,'timings':[{'index':i,'operation_first':i==0,'operation_memo_warm':i>0,'nanoseconds':12}for i in range(2)],'OS_page_cache_cold_claimed':False,'cache_drop_or_model_inference_or_GPU_execution':False,'serving_latency_or_speed_qualified':False};r.trial_metadata(row,'B',1)
  for key,val in [('nanoseconds',True),('index',True),('operation_first',1)]:
   bad=copy.deepcopy(row);bad['timings'][0][key]=val;self.assertRaises(ValueError,r.trial_metadata,bad,'B',1)
  bad=copy.deepcopy(row);bad['finished_epoch']=float('inf');self.assertRaises(ValueError,r.trial_metadata,bad,'B',1)
 def test_full_scope_cannot_be_numeric_boolean(self):
  self.assertRaises(ValueError,r.finalized_binding,'/not-read',full_bytes=1)
if __name__=='__main__':unittest.main()
