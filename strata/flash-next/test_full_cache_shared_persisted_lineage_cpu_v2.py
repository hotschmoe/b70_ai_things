import copy,json,struct,tempfile,threading,types,unittest
from pathlib import Path
from full_cache_shared_persisted_v2 import header_hash56
from full_cache_shared_persisted_lineage_v2 import negative_file,session_binding
from full_cache_shared_session_observer_v2 import install

class Sessions(unittest.TestCase):
 def test_new_negative_file_only_changes_original_header_fingerprint(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);header=b'STRSESS\x01'+struct.pack('<IIQQQQQ',1,64,7,9,16,0,0);header+=struct.pack('<Q',header_hash56(header));original=root/'valid.bin';original.write_bytes(header+b'p'*16+b't'*16);raw=original.read_bytes();result=negative_file(original,root/'wrong.bin');self.assertEqual(original.read_bytes(),raw);self.assertEqual((root/'wrong.bin').read_bytes()[64:],raw[64:]);self.assertFalse(result['actual_restore_execution_observed'])
 def fixture(self,negative=False):
  owner={'engine_pid':17,'engine_generation':1};action='restore' if negative else 'save';filename='wrong.bin' if negative else 'valid.bin';path='/results/sessions/'+filename
  begin=dict(owner,kind='session_begin',sequence=1,call=3,action=action,path=path);end=dict(owner,kind='session_end',sequence=4,call=3,action=action,path=path,result={'tokens':2,'bytes':96,'ms':1.0},error=None);terminal='SAVED 2 96 1.0';body={'id_slot':0,'filename':filename,'n_saved':2,'n_written':96};status=200
  if negative:
   reason='session file: saved with another model (model fingerprint differs)';terminal='SERR invalid 0 '+reason;end.update(result=None,error={'kind':'invalid','published':False,'message':reason});body={'error':{'code':400,'type':'invalid_request_error','kind':'invalid','message':reason}};status=400
  events=[begin,dict(owner,kind='native_send',sequence=2,call=3,line=('RESTORE ' if negative else 'SAVE ')+path),dict(owner,kind='native_receive',sequence=3,line=terminal),end];http={'request':{'method':'POST','path':'/slots/0?action='+action,'body':{'filename':filename}},'error':None,'status':status,'body':body};expected={'action':'restore_wrong_model' if negative else 'save','filename':filename,'expected_saved_tokens':2};return events,http,expected,owner
 def test_actual_HTTP_native_save_and_unpublished_fingerprint_refusal(self):
  for negative in (False,True):
   self.assertTrue(session_binding(*self.fixture(negative))['actual_FIFO_native_and_HTTP_session_joined'])
 def test_foreign_source_path_checksum_disabled_refusal_rejected(self):
  for mutate in ('pid','path','checksum','disabled','published'):
   events,http,expected,owner=self.fixture(True)
   if mutate=='pid':events[2]['engine_pid']=18
   elif mutate=='path':events[1]['line']='RESTORE /results/sessions/foreign.bin'
   elif mutate=='checksum':events[2]['line']='SERR invalid 0 header checksum differs'
   elif mutate=='disabled':http.update(status=501,body={'error':{'code':501,'type':'server_error','message':'slot save/restore is disabled'}})
   else:events[-1]['error']['published']=True
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):session_binding(events,http,expected,owner)
 def test_original_session_observer_does_not_modify_command_or_return(self):
  calls=[]
  class Refused(Exception):pass
  class Engine:
   batch=0;gen=1;proc=types.SimpleNamespace(pid=17)
   def session_file(self,action,path):calls.append((action,path));return {'tokens':2,'bytes':96,'ms':1.0}
  server=types.SimpleNamespace(StrataEngine=Engine,SessionRefused=Refused);events=[];active=set();local=threading.local();install(server,events.append,local,[0],threading.Lock(),active);result=Engine().session_file('save','/results/sessions/valid.bin');self.assertEqual(result,{'tokens':2,'bytes':96,'ms':1.0});self.assertEqual(calls,[('save','/results/sessions/valid.bin')]);self.assertFalse(active);self.assertIsNone(local.call);self.assertEqual([e['kind']for e in events],['session_begin','session_end'])
if __name__=='__main__':unittest.main()
