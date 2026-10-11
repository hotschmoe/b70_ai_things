"""Tiny metadata-only saved postACK source/page joins."""
import copy,tempfile,types,unittest
from pathlib import Path
import full_cache_shared_health_handoff_v9 as h

class Seal(unittest.TestCase):
 def test_current_pages_keep_original_epoch_and_all_other_fields(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);receipt=root/'identity.json';receipt.write_text('{}');page={'epoch':12.0,'path':'/tiny/third','passed':True,'rows':[{'offset':4096,'sha256':'a'*64}],'stat_before':[1,2,8192,4,5]};identity={'passed':True,'rows':['tiny-four-roster'],'current_known_pages':page};seal={'started_epoch':11.0,'finished_epoch':13.0,'current_source_binding':{'source':40},'current_model_identity_and_pages':copy.deepcopy(identity)};current=copy.deepcopy(identity);current['current_known_pages']['epoch']=99.0;ctrl=types.SimpleNamespace(HERE=root,ROOT=root,source_binding=lambda:{'source':40},read=lambda p:{'destination':'tiny','files':[{'path':'UD-Q4_K_XL/'+str(i)}for i in range(4)]},verify_model_identity=lambda *args:current);plan={'model_identity':{'path':str(receipt),'sha256':h.sha(receipt)}}
   self.assertTrue(h.seal_context_binding(seal,plan,ctrl));self.assertEqual(seal['current_model_identity_and_pages']['current_known_pages']['epoch'],12.0)
   current['current_known_pages']['rows'][0]['sha256']='b'*64
   with self.assertRaises(ValueError):h.seal_context_binding(seal,plan,ctrl)
 def test_original_pages_before_seal_or_current_source_change_fail(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);receipt=root/'identity';receipt.write_text('{}');identity={'passed':True,'current_known_pages':{'epoch':9.0,'rows':[]}};seal={'started_epoch':11,'finished_epoch':13,'current_source_binding':{'source':40},'current_model_identity_and_pages':identity};ctrl=types.SimpleNamespace(HERE=root,ROOT=root,source_binding=lambda:{'source':40},read=lambda p:{'destination':'tiny','files':[]},verify_model_identity=lambda *args:identity);plan={'model_identity':{'path':str(receipt),'sha256':h.sha(receipt)}}
   with self.assertRaises(ValueError):h.seal_context_binding(seal,plan,ctrl)
   ctrl.source_binding=lambda:{'source':39}
   with self.assertRaises(ValueError):h.seal_context_binding(seal,plan,ctrl)
if __name__=='__main__':unittest.main()
