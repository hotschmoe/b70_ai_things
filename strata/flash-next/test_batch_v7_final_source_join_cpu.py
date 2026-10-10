#!/usr/bin/env python3
"""Tiny synthetic final source/path/chronology/plan/page controls, no real weights."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit_batch_numerical_suite_v7 as q
import batch_numerical_proofs_v7 as proof
import batch_numerical_execution_v7 as ctrl
import source_page_watchdog_v3 as pages

class FinalSourceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='batch7-final-source-CPU-');self.root=Path(self.tmp.name);self.directory=self.root/'run';(self.directory/'child').mkdir(parents=True);self.model=self.root/'tiny-model';self.model.mkdir();self.here=self.root/'helper';self.here.mkdir();self.files=[];self.rows=[];self.shards=[]
  for n in range(4):
   path=self.model/('part%d.gguf'%n);path.write_bytes(bytes([n+1])*8192);stat=path.stat();sig=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns];digest=q.sha(path);self.files.append({'path':path.name,'size':8192,'sha256':digest});self.rows.append({'path':str(path),'passed':True,'bytes':8192,'sha256':digest,'expected_sha256':digest,'stat_before':sig,'stat_after':sig});self.shards.append(path)
  # Real production files are selected by UD prefix. Syntheticrelative names keepthat contract.
  for f in self.files:f['path']='UD-Q4_K_XL/'+f['path']
  sub=self.model/'UD-Q4_K_XL';sub.mkdir()
  for n,path in enumerate(self.shards):path.rename(sub/path.name);self.shards[n]=sub/path.name;self.rows[n]['path']=str(self.shards[n]);st=self.shards[n].stat();self.rows[n]['stat_before']=self.rows[n]['stat_after']=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  self.lock={'destination':str(self.model),'revision':'CPU_SYNTHETIC','files':self.files};self.write(self.here/'model-lock.json',self.lock);self.chain={'CPU_GENUINE_CHAIN_MOCK_ONLY':True};self.prepared={'engine_receipt_sha256':'a'*64,'engine_receipt':'CPU_SYNTHETIC_SDK/receipt.json','cards':[0]};self.plan={'prepared':'CPU_SYNTHETIC_C113','lane':'source35','engine_root':'CPU_SYNTHETIC_SDK','engine_receipt_sha256':'a'*64,'cards':[0],'baseline_source_proof':self.chain};self.external=self.root/'external-plan.json'
  for path in (self.external,self.directory/'input-plan.snapshot.json',self.directory/'child/plan.snapshot.json'):self.write(path,self.plan)
  self.identity={'passed':True,'lock_sha256':q.sha(self.here/'model-lock.json'),'model_revision':self.lock['revision'],'started':3.,'finished':4.,'after_child_terminal_epoch':2.,'rows':self.rows};self.write(self.directory/'post-model-identity.json',self.identity);self.health={'passed':True,'finished_epoch':2.};self.write(self.directory/'post-health.json',self.health)
  self.current={'passed':True,'path':str(self.shards[2]),'stat_before':self.rows[2]['stat_before'],'stat_after':self.rows[2]['stat_after'],'epoch':7.,'rows':[]};self.expected=[]
  for offset in (0,4096):
   data=self.shards[2].read_bytes()[offset:offset+4096];digest=hashlib.sha256(data).hexdigest();self.expected.append((offset,digest));self.current['rows'].append({'offset':offset,'bytes':4096,'sha256':digest,'expected_sha256':digest,'passed':True})
  self.parent={'plan':str(self.external),'plan_sha256':q.sha(self.external),'post_model_identity':{'path':str(self.directory/'post-model-identity.json'),'sha256':q.sha(self.directory/'post-model-identity.json')},'prepared_chain':self.chain,'post_health_finished_epoch':2.,'child_terminal_epoch':1.5,'finished_epoch':6.}
  for name,epoch in [('known_pages_before_hash',2.5),('known_pages_after_hash',5.)]:
   saved=copy.deepcopy(self.current);saved['epoch']=epoch
   for row in saved['rows']:
    path=self.directory/(name+'-'+str(row['offset'])+'.raw');path.write_bytes(self.shards[2].read_bytes()[row['offset']:row['offset']+4096]);row['preserved_path']=str(path)
   self.parent[name]=saved
  self.child={'plan_sha256':q.sha(self.external),'finished_epoch':1.}
  self.patches=[patch.object(q,'HERE',self.here),patch.object(proof,'genuine_baseline',lambda *args:(self.prepared,self.chain)),patch.object(proof,'engine_binding',lambda *args:{}),patch.object(ctrl,'verify_model_identity',lambda *args:{'CPU_MOCK_CURRENT_VERIFIED':True}),patch.object(pages,'KNOWN_PAGES',tuple(self.expected)),patch.object(pages,'guard',lambda *args:self.current)]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.tmp.cleanup()
 def write(self,path,value):path.write_text(json.dumps(value,indent=2)+'\n')
 def run_join(self):return q.final_source_join(self.directory,self.parent,self.plan,self.child)
 def update_identity(self):self.write(self.directory/'post-model-identity.json',self.identity);self.parent['post_model_identity']['sha256']=q.sha(self.directory/'post-model-identity.json')
 def test_positive_current_join_bothpages_new4_and_snapshot(self):self.assertTrue(self.run_join()['new_full4_bracketed_by_two_page_views'])
 def test_child_input_external_plan_content_or_sha_mismatch(self):
  for path in (self.external,self.directory/'input-plan.snapshot.json',self.directory/'child/plan.snapshot.json'):
   old=path.read_bytes();self.write(path,dict(self.plan,changed=True))
   with self.assertRaises(ValueError):self.run_join()
   path.write_bytes(old)
  self.child['plan_sha256']='b'*64
  with self.assertRaises(ValueError):self.run_join()
 def test_postidentity_foreign_path_and_hash_rejected(self):
  other=self.root/'foreign.json';other.write_bytes((self.directory/'post-model-identity.json').read_bytes());self.parent['post_model_identity']['path']=str(other)
  with self.assertRaises(ValueError):self.run_join()
 def test_new4_before_native_terminal_or_posthealth_rejected(self):
  for key,value in [('started',1.9),('after_child_terminal_epoch',1.5),('finished',2.9)]:
   old=self.identity[key];self.identity[key]=value;self.update_identity()
   with self.assertRaises(ValueError):self.run_join()
   self.identity[key]=old;self.update_identity()
 def test_modelrevision_shard_order_path_and_stat_rejected(self):
  original=copy.deepcopy(self.identity)
  for mutation in ('revision','order','path','stat'):
   self.identity=copy.deepcopy(original)
   if mutation=='revision':self.identity['model_revision']='OTHER'
   elif mutation=='order':self.identity['rows'][0],self.identity['rows'][1]=self.identity['rows'][1],self.identity['rows'][0]
   elif mutation=='path':self.identity['rows'][0]['path']=str(self.root/'WRONG_SOURCE')
   else:self.identity['rows'][0]['stat_after'][-1]+=1
   self.update_identity()
   with self.assertRaises(ValueError):self.run_join()
 def test_current_file_stat_change_rejected(self):
  self.shards[0].write_bytes(self.shards[0].read_bytes()+b'X')
  with self.assertRaises(ValueError):self.run_join()
 def test_c113_chain_or_actual_SDK_association_rejected(self):
  self.parent['prepared_chain']={'OLD_C113':True}
  with self.assertRaises(ValueError):self.run_join()
 def test_foreign_currentSDK_path_rejected(self):
  self.prepared['engine_receipt']='OTHER_SDK/receipt.json'
  with self.assertRaises(ValueError):self.run_join()
 def test_missing_or_false_pageproof_or_wrong_offsets_rejected(self):
  original=copy.deepcopy(self.parent['known_pages_before_hash'])
  for mutation in ('failed','missing','offset'):
   self.parent['known_pages_before_hash']=copy.deepcopy(original)
   if mutation=='failed':self.parent['known_pages_before_hash']['passed']=False
   elif mutation=='missing':self.parent['known_pages_before_hash']['rows'].pop()
   else:self.parent['known_pages_before_hash']['rows'][0]['offset']=1
   with self.assertRaises(ValueError):self.run_join()
 def test_page_bracket_chronology_rejected(self):
  self.parent['known_pages_before_hash']['epoch']=3.1
  with self.assertRaises(ValueError):self.run_join()
 def test_preserved_page_changed_or_escaped_rejected(self):
  row=self.parent['known_pages_after_hash']['rows'][0];path=Path(row['preserved_path']);path.write_bytes(b'X'*4096)
  with self.assertRaises(ValueError):self.run_join()
if __name__=='__main__':unittest.main()
