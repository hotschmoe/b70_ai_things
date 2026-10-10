import copy,json,unittest
from full_cache_shared_batch0_pcl_v2 import request_view
from full_cache_shared_history_v2 import digest

class Lineage(unittest.TestCase):
 def fixture(self):
  ids=[11,12];begin={'kind':'engine_begin','sequence':1,'call':3,'engine_pid':17,'engine_generation':1,'submitted_ids':ids,'submitted_ids_sha256':digest(ids),'rendered_matches_submitted':True,'rendered_prompt':{'ids':ids},'max_new':1,'embeddings':None};end={'kind':'engine_end','sequence':7,'call':3,'engine_pid':17,'engine_generation':1,'error':None,'cancelled':False,'generated_ids':[9],'generated_ids_sha256':digest([9])};pclbegin={'event':'begin','pid':17,'request':1,'tokens':2,'input_sha256_le32':digest(ids)};commit={'event':'committed_live','pid':17,'request':1,'ids':ids,'tokens':2,'sha256_le32':digest(ids),'ids_truncated':False,'phase':'complete'}
  events=[begin,{'kind':'native_send','sequence':2,'call':3,'engine_pid':17,'line':'GEN 1 fresh=1 11,12'},{'kind':'native_receive','sequence':3,'engine_pid':17,'line':'PCL '+json.dumps(pclbegin)},{'kind':'native_receive','sequence':4,'engine_pid':17,'line':'T 9'},{'kind':'native_receive','sequence':5,'engine_pid':17,'line':'DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2'},end,{'kind':'native_receive','sequence':8,'engine_pid':17,'line':'PCL '+json.dumps(commit)}];return events,begin,end,{'ids':ids,'fresh':1,'pin':None,'expected_reused':0}
 def test_actual_FIFO_and_late_PCL_commit_ordinal_no_fake_rid(self):
  events,begin,end,expected=self.fixture();view,binding=request_view(events,begin,end,expected);self.assertEqual(view['output_ids'],[9]);self.assertEqual(binding['actual_native_PCL_request_ordinal'],1);self.assertFalse(binding['strict_batch_RID_claimed']);self.assertFalse(binding['original_records_rewritten'])
 def test_other_actor_or_HTTP_output_cannot_borrow_native_ordinal(self):
  for change in ('pid','token','call','rid','other_begin'):
   events,begin,end,expected=self.fixture()
   if change=='pid':events[2]['engine_pid']=18
   elif change=='token':end['generated_ids']=[10];end['generated_ids_sha256']=digest([10])
   elif change=='call':events[1]['call']=4
   elif change=='rid':events[1]['line']='GEN 1 fresh=1 rid=3 11,12'
   else:events.insert(2,dict(begin,sequence=2.5,call=4))
   with self.subTest(change=change),self.assertRaises(ValueError):request_view(events,begin,end,expected)
 def test_emitted_unconsumed_token_cannot_be_saved_prefix(self):
  events,begin,end,expected=self.fixture();commit=json.loads(events[-1]['line'][4:]);commit.update(ids=[11,12,9],tokens=3,sha256_le32=digest([11,12,9]));events[-1]['line']='PCL '+json.dumps(commit)
  with self.assertRaises(ValueError):request_view(events,begin,end,expected)
 def test_actual_PCL_numeric_types_and_incomplete_roster_rejected(self):
  for change in ('float_pid','float_request','missing_commit','wrong_digest'):
   events,begin,end,expected=self.fixture()
   if change=='missing_commit':events.pop()
   else:
    row=json.loads(events[2]['line'][4:]);row.update(**({'pid':17.0} if change=='float_pid' else {'request':1.0} if change=='float_request' else {'input_sha256_le32':'a'*64}));events[2]['line']='PCL '+json.dumps(row)
   with self.subTest(change=change),self.assertRaises(ValueError):request_view(events,begin,end,expected)
 def test_expected_policy_cannot_follow_observed_reuse(self):
  events,begin,end,expected=self.fixture();expected['expected_reused']=1
  with self.assertRaises(ValueError):request_view(events,begin,end,expected)
if __name__=='__main__':unittest.main()
