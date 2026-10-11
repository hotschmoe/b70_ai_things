import copy,json,unittest
from full_cache_shared_history_v2 import digest
import full_cache_shared_batch0_decode_v9 as d

class Decode(unittest.TestCase):
 def fixture(self):
  messages=[{'role':'user','content':'owned'}];begin={'kind':'engine_begin','call':3,'engine_pid':17,'engine_generation':1,'rendered_prompt':{'messages':messages}};end={'kind':'engine_end','call':3,'engine_pid':17,'engine_generation':1,'rid':None,'generated_ids':[11],'generated_ids_sha256':digest([11]),'cancelled':False};client={'request':{'messages':messages},'done_received':True,'cancel_requested':False,'error':None,'events':[{'data':json.dumps({'choices':[{'delta':{'content':'word'}}]})},{'data':'[DONE]'}]};lineage={'actual_native_PCL_pid':17,'actual_HTTP_call':3,'actual_native_PCL_request_ordinal':1,'strict_batch_RID_claimed':False};return messages,client,begin,end,lineage
 def test_actual_PCL_ordinal_used_without_mutating_missing_nativeRID(self):
  values=self.fixture();request,source=d.decode_request(*values);self.assertEqual(request['case'],3);self.assertEqual(request['repeat'],1);self.assertIsNone(source['native_end']['rid']);self.assertFalse(source['numeric_lineage']['strict_batch_RID_claimed'])
 def test_new_original_decode_exact_text_and_ids(self):
  request,source=d.decode_request(*self.fixture());hashes={'vocab.json':'a'*64};observed={'actual_GPU_touch':False,'actual_model_payload_read':False,'fixtures':[],'tokenizer_file_sha256':hashes,'decoded_outputs':[{'case':3,'repeat':1,'text':'word','matches':True}]};result=d.decoded_binding(request,source,observed,hashes);self.assertEqual(result['original_generated_ids'],[11]);self.assertTrue(result['independent_original_tokenizer_decode']);self.assertFalse(result['strict_batch_RID_claimed'])
  observed['decoded_outputs'][0]['text']='borrowed'
  with self.assertRaises(ValueError):d.decoded_binding(request,source,observed,hashes)
 def test_other_actor_or_virtualRID_or_incomplete_HTTP_refused(self):
  for key,value in [('actual_native_PCL_pid',18),('actual_HTTP_call',4),('actual_native_PCL_request_ordinal',1.0),('strict_batch_RID_claimed',True)]:
   values=self.fixture();values[-1][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):d.decode_request(*values)
 def test_natural_EOS_empty_text_can_be_independently_decoded(self):
  values=self.fixture();values[1]['events']= [{'data':'[DONE]'}];request,source=d.decode_request(*values);self.assertEqual(request['content'],'')
if __name__=='__main__':unittest.main()
