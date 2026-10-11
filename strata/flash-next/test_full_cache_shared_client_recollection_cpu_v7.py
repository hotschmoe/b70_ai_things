import copy,json,tempfile,unittest
from pathlib import Path
from full_cache_shared_client_recollection_v7 import recollect

class Client(unittest.TestCase):
 def fixture(self,root):
  directory=root/'prime'/'client';directory.mkdir(parents=True);rid='full-cache-prime-client-1-0';messages=[[{'role':'user','content':'owned input'}]];body={'model':'hotschmoe-dd','messages':messages[0],'stream':True,'temperature':0,'max_tokens':1,'frequency_penalty':0,'presence_penalty':0,'user':rid,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'strata_fresh':True};row={'request_id':rid,'client_index':0,'request':body,'sent_epoch':1.,'finished_epoch':3.,'error':None,'http_status':200,'cancel_requested':False,'done_received':True,'events':[{'epoch':2.,'data':json.dumps({'choices':[{'delta':{'content':'reply'}}]})},{'epoch':2.5,'data':'[DONE]'}]};saved={'rows':[row],'requested_streams':1};(directory/'models.json').write_text(json.dumps({'data':[{'id':'hotschmoe-dd'},{'id':'alias'}]}));self.save(directory,saved);return directory,saved,messages
 def save(self,directory,saved):
  (directory/'client.json').write_text(json.dumps(saved));(directory/(saved['rows'][0]['request_id']+'.json')).write_text(json.dumps(saved['rows'][0]))
 def test_original_individual_packet_and_SSE_roster(self):
  with tempfile.TemporaryDirectory() as td:
   d,s,m=self.fixture(Path(td));result=recollect(d,s,m,[1],{'strata_fresh':True},'alias');self.assertEqual(result['HTTP_texts'],['reply']);self.assertTrue(result['independent_native_ID_to_text_decode_still_required'])
 def test_typed_body_and_original_event_epoch_rejected(self):
  for kind in ('body','epoch','tail'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
    d,s,m=self.fixture(Path(td));r=s['rows'][0]
    if kind=='body':r['request']['max_tokens']=True
    elif kind=='epoch':r['events'][0]['epoch']=4.
    else:r['events'].append({'epoch':2.8,'data':'{}'})
    self.save(d,s)
    with self.assertRaises(ValueError):recollect(d,s,m,[1],{'strata_fresh':True},'alias')
 def test_changed_individual_copy_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   d,s,m=self.fixture(Path(td));p=d/(s['rows'][0]['request_id']+'.json');raw=json.loads(p.read_bytes());raw['done_received']=False;p.write_text(json.dumps(raw))
   with self.assertRaises(ValueError):recollect(d,s,m,[1],{'strata_fresh':True},'alias')
if __name__=='__main__':unittest.main()
