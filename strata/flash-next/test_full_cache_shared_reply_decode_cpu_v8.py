import copy,json,unittest
import full_cache_shared_reply_decode_v8 as d

class Reply(unittest.TestCase):
 def fixture(self,cancel=False):
  messages=[{'role':'user','content':'own useful question'}];client={'request':{'messages':messages},'done_received':not cancel,'cancel_requested':cancel,'error':None,'events':[{'data':json.dumps({'choices':[{'delta':{'content':'wor' if cancel else 'word'}}]})}]+([] if cancel else [{'data':'[DONE]'}])};begin={'kind':'engine_begin','sequence':2,'call':1,'engine_pid':17,'engine_generation':1,'rendered_prompt':{'messages':messages}};end={'kind':'engine_end','sequence':4,'call':1,'rid':3,'engine_pid':17,'engine_generation':1,'generated_ids':[11],'generated_ids_sha256':d.digest([11]),'cancelled':cancel};phase={'phase':{'name':'actual','rows':[{'messages':messages}]},'client':{'rows':[client]},'work':[{'call':1,'rid':3}],'producer_ack':{'sequence':1},'end_sequence':5};return [phase],[begin,end]
 def observed(self,text='word',matches=True):return {'actual_GPU_touch':False,'actual_model_payload_read':False,'fixtures':[],'tokenizer_file_sha256':{'vocab.json':'a'*64},'decoded_outputs':[{'case':1,'repeat':3,'text':text,'matches':matches}]}
 def test_all_complete_replies_exact_own_ID_text(self):
  requests,records=d.batch_requests(*self.fixture());proof=d.decoded_binding(requests,records,self.observed(),{'vocab.json':'a'*64});self.assertTrue(proof['original_API_yielded_ID_HTTP_text_qualified']);self.assertFalse(proof['native_accepted_ID_roster_authority_borrowed_from_API_yields'])
 def test_cancelled_delivery_prefix_is_explicit_not_full_token_delivery(self):
  requests,records=d.batch_requests(*self.fixture(True));proof=d.decoded_binding(requests,records,self.observed(matches=False),{'vocab.json':'a'*64});self.assertEqual(proof['actual_all_reply_decodes'][0]['mode'],'intentional_cancel_delivered_text_prefix');self.assertFalse(proof['actual_cancelled_HTTP_delivered_token_ID_roster_observed'])
  with self.assertRaises(ValueError):d.decoded_binding(requests,records,self.observed('borrowed',False),{'vocab.json':'a'*64})
 def test_wrong_RID_or_missing_reply_cannot_borrow_decode(self):
  phases,events=self.fixture();events[1]['rid']=4
  with self.assertRaises(ValueError):d.batch_requests(phases,events)
  phases,events=self.fixture();phases[0]['client']['rows']=[]
  with self.assertRaises(ValueError):d.batch_requests(phases,events)
 def test_returned_false_or_different_complete_text_rejected(self):
  requests,records=d.batch_requests(*self.fixture())
  for text,match in [('word',False),('borrowed',False)]:
   with self.assertRaises(ValueError):d.decoded_binding(requests,records,self.observed(text,match),{'vocab.json':'a'*64})
