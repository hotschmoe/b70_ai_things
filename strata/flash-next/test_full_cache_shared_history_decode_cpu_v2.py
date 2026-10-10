import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import test_full_cache_shared_history_cpu_v2 as fixture
import full_cache_shared_history_decode_v2 as d

class Decode(unittest.TestCase):
 def inputs(self):
  f=fixture.History();f.setUp();return d.decode_request(f.messages,f.client,f.begin,f.end)
 def test_new_decode_joins_own_native_IDs_and_HTTP_text(self):
  request,source=self.inputs();hashes={'vocab.json':'a'*64};observed={'actual_GPU_touch':False,'actual_model_payload_read':False,'fixtures':[],'tokenizer_file_sha256':hashes,'decoded_outputs':[{'case':request['case'],'repeat':request['repeat'],'text':request['content'],'matches':True}]}
  before=copy.deepcopy(source);result=d.decoded_binding(request,source,observed,hashes)
  self.assertEqual(source,before);self.assertTrue(result['original_mechanism']['mechanism_only_no_old_output_proof_transfer'])
  for key,value in [('fixtures',[{}]),('actual_GPU_touch',True),('tokenizer_file_sha256',{'vocab.json':'b'*64})]:
   bad=copy.deepcopy(observed);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):d.decoded_binding(request,source,bad,hashes)
  bad=copy.deepcopy(observed);bad['decoded_outputs'][0]['repeat']+=1
  with self.assertRaises(ValueError):d.decoded_binding(request,source,bad,hashes)
 def test_foreign_actor_or_changed_reply_refused_before_decode(self):
  f=fixture.History();f.setUp();f.begin['engine_pid']=11
  with self.assertRaises(ValueError):d.decode_request(f.messages,f.client,f.begin,f.end)
 def test_exact_original_runner_metadata_only_mounts(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'source';pack=root/'pack';output=root/'output';source.mkdir();pack.mkdir();output.mkdir()
   (pack/'original.gguf').write_bytes(b'payload must never be mounted')
   command=d.command_recipe(output,source,pack,d.IMAGE,'a'*64,'b70-prefix-17-shared39-history-decode-test','b'*64)
   self.assertNotIn('--device',command);self.assertNotIn('--gpus',command);self.assertEqual(command[command.index('--network')+1],'none');self.assertEqual(command[-1],'/results/output-id-requests.json')
   mounts=[command[i+1] for i,a in enumerate(command) if a=='-v'];self.assertFalse(any(m.startswith(str(pack)+':') for m in mounts));self.assertFalse(any('original.gguf' in m for m in mounts));self.assertEqual(sum('/tokenizer/' in m for m in mounts),5)
   self.assertIn(str(d.RUNNER)+':'+d.CONTAINER_RUNNER+':ro',mounts)
   with self.assertRaises(ValueError):d.command_recipe(output,source,pack,'other-image','a'*64,'b70-prefix-17-shared39-history-decode-test','b'*64)
 def test_prepared_pack_root_has_declared_tokenizer_subdirectory(self):
  expected={name:'a'*64 for name in ('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')}
  root,hashes=d.tokenizer_location({'pack':'/owned/model-pack'},{'pack':'/owned/model-pack'},{'tokenizer_files':expected})
  self.assertEqual(root,Path('/owned/model-pack/tokenizer'));self.assertEqual(hashes,expected)
  with self.assertRaises(ValueError):d.tokenizer_location({'pack':'/other'},{'pack':'/owned/model-pack'},{'tokenizer_files':expected})
 def test_late_creation_not_absence_and_forced_stop_recorded(self):
  with tempfile.TemporaryDirectory() as temp:
   terminal={'Name':'/owned','State':{'Running':False}};running={'Name':'/owned','State':{'Running':True}};seen=iter([None,running,terminal,None]);commands=[];ledger={'retirement_failures':[],'forced_owned_stop':False}
   def owned(obj):self.assertEqual(obj['Name'],'/owned')
   def run(command,**kwargs):commands.append(command);return SimpleNamespace(returncode=0)
   result=d.retire_owned_metadata('owned',owned,temp,ledger,lambda name:next(seen),run,lambda delay:None)
   self.assertEqual(result,terminal);self.assertTrue(ledger['owned_terminal_and_removed']);self.assertTrue(ledger['forced_owned_stop']);self.assertEqual(len(ledger['retirement_failures']),1);self.assertEqual([c[1] for c in commands],['stop','rm'])
 def test_foreign_unknown_owner_preserves_failure_not_removal(self):
  with tempfile.TemporaryDirectory() as temp:
   ledger={'retirement_failures':[]};commands=[]
   def owned(obj):raise ValueError('Foreign label')
   def pause(delay):raise RuntimeError('test ends unresolved hold')
   with self.assertRaises(RuntimeError):d.retire_owned_metadata('owned',owned,temp,ledger,lambda name:{'State':{'Running':False}},lambda *a,**k:commands.append(a),pause)
   self.assertTrue(ledger['retirement_failures']);self.assertEqual(commands,[]);self.assertNotIn('owned_terminal_and_removed',ledger)

if __name__=='__main__':unittest.main()
