"""Exercise actual frozen extract lifecycle derivation with CPU numeric fixture."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import serial_prefix_qualification_v9 as v9
from serial_request_recollection_v1 import recollect_request

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.path=self.root/'request.json';ids=[1,2];committed=ids+[9]
  events=[{'event':'begin','pid':1,'request':1,'tokens':2,'input_sha256_le32':v9.le32_digest(ids)}]
  for complete in [False,True]:events.append({'event':'stage_span','pid':1,'request':1,'phase':'verify_body','device':0,'lb':0,'le':48,'lo':0,'hi':2,'complete':complete})
  events.append({'event':'committed_live','pid':1,'request':1,'ids':committed,'tokens':3,'ids_truncated':False,'sha256_le32':v9.le32_digest(committed),'published':True,'chain_updated':True,'live_reusable':True,'phase':'complete'})
  self.raw={'label':'independent-context','ids':ids,'fresh':0,'pin':0,'command':v9.request_command(ids,64,0,0),'stderr':['PCL '+json.dumps(e) for e in events]}
  self.numeric={'ledger':{'cancelled':False,'actual_reused':0,'evaluated_prompt_rows':2},'selection':{'candidate_resume':0,'actual_reused':0,'stages':1},'spans':[],'vectors':[],'logits':[],'residuals':[]}
  self.path.write_text(json.dumps(self.raw));self.savedraw=copy.deepcopy(self.raw)
  with patch.object(v9,'extract_numeric',return_value=copy.deepcopy(self.numeric)):self.savedmeta=v9.extract(self.savedraw,self.root,True,True,[(0,48)])
  self.saved={'raw':self.savedraw,'meta':self.savedmeta};self.original_bytes=self.path.read_bytes()
 def recollect(self):
  with patch.object(v9,'extract_numeric',return_value=copy.deepcopy(self.numeric)):return recollect_request(self.path,self.saved,self.root,[(0,48)],v9.extract)
 def test_actual_extract_adds_committed_ids_before_complete_equality(self):
  self.assertNotIn('lifecycle_committed_ids',self.raw);self.assertEqual(self.savedraw['lifecycle_committed_ids'],[1,2,9]);derived,meta=self.recollect();self.assertEqual(derived,self.savedraw);self.assertEqual(meta,self.savedmeta);self.assertEqual(self.path.read_bytes(),self.original_bytes);self.assertNotIn('lifecycle_committed_ids',self.raw)
 def test_wrong_saved_derived_ids_rejected(self):
  self.saved['raw']['lifecycle_committed_ids']=[1,2,10]
  with self.assertRaisesRegex(ValueError,'derived raw'):self.recollect()
 def test_foreign_raw_field_not_ignored(self):
  changed=copy.deepcopy(self.raw);changed['unexplained']='mutation';self.path.write_text(json.dumps(changed))
  with self.assertRaisesRegex(ValueError,'derived raw'):self.recollect()
 def test_changed_stored_metadata_rejected(self):
  self.saved['meta']['ledger']['evaluated_prompt_rows']=1
  with self.assertRaisesRegex(ValueError,'metadata'):self.recollect()
 def test_changed_source_lifecycle_not_normalized_away(self):
  changed=copy.deepcopy(self.raw);changed['stderr'][0]=changed['stderr'][0].replace(v9.le32_digest([1,2]),'wrong');self.path.write_text(json.dumps(changed))
  with self.assertRaises(ValueError):self.recollect()
if __name__=='__main__':unittest.main()
