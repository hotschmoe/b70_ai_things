#!/usr/bin/env python3
"""CPU metadata/synthetic controls only; no real model/GPU/qualification."""
import ast,copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import qualify_c1_serving_combined_v4 as r

class FinalTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.signature=[1,2,100,4,5]
  self.lock={'revision':'CPU_mock','destination':'models/mock','files':[{'path':'UD-Q4_K_XL/mock-0000%d-of-00004.gguf'%i,'sha256':str(i)*64,'size':100} for i in range(1,5)]}
  rows=[{'path':str(r.ROOT/self.lock['destination']/f['path']),'passed':True,'sha256':f['sha256'],'expected_sha256':f['sha256'],'bytes':100,'stat_before':self.signature,'stat_after':self.signature} for f in self.lock['files']]
  self.identity={'passed':True,'rows':rows,'started':9.1,'finished':12.2,'after_terminal_and_post_health_epoch':8.2,'lock_sha256':'lock','model_revision':'CPU_mock'}
  third=next(row['path'] for row in rows if '00003-of-00004' in row['path'])
  def guard(epoch):return {'passed':True,'path':third,'stat_before':self.signature,'stat_after':self.signature,'epoch':epoch,'rows':[{'offset':offset,'bytes':4096,'sha256':sha,'expected_sha256':sha,'passed':True} for offset,sha in r.watchdog.KNOWN_PAGES]}
  self.proof={'passed':True,'parent_controller_sha256':'parent','controller_sha256':r.CONTROLLER_SHA,'watchdog_sha256':r.WATCHDOG_SHA,'prepared_sha256':'prepared','engine_receipt_sha256':'engine','upload_v2_post_identity_sha256':'uploadpost','terminal_epoch':7.2,'post_health_finished_epoch':8.2,'post_health_sha256':'health','identity':{'path':str(self.root/'c1-post-model-identity-v4.json'),'sha256':'identity'},'known_pages_before_hash':guard(8.9),'known_pages_after_hash':guard(12.3),'page_guard_calls':3,'between_polls_identity_observed':False}
  self.final={'passed':True,'teardown_passed':True,'post_health_passed':True,'c1_parent_generation':4,'post_full4_source_qualified':True,'engine_receipt_sha256':'engine','post_health_sha256':'health','c1_parent_controller_sha256':'parent','controller_qualification_sha256':'raw','c1_source_identity_proof':{'path':str(self.root/'c1-source-identity-proof-v4.json'),'sha256':'proof'}}
  self.prepared={'engine_receipt_sha256':'engine','upload_lifecycle':{'post_model_identity_sha256':'uploadpost','post_full4_source_qualified':True}}
  self.parent={'passed':True,'parent_generation':4,'parent_controller_sha256':'parent','owned_terminal':True,'post_health_passed':True,'launch_supervisor_exit_code':0,'qualification_sha256':'final','source_identity_proof_sha256':'proof','engine_receipt_sha256':'engine','post_model_identity':{'sha256':'identity','passed':True},'owned_terminal_epoch':7.2,'post_health_finished_epoch':8.2,'finished_epoch':13.}
  self.health={'passed':True,'finished_epoch':8.2,'files':[{'path':'mock-health','sha256':'healthlog'},{'path':'mock-collective','sha256':'collectivelog'}]}
 def validate(self):
  read={str(self.root/'parent-qualification.json'):self.parent,str(self.root/'qualification.json'):self.final,str(self.root/'c1-source-identity-proof-v4.json'):self.proof,str(self.root/'c1-post-model-identity-v4.json'):self.identity,str(self.root/'post-health.json'):self.health,str(r.ROOT/'strata/flash-next/model-lock.json'):self.lock}
  hashes={str(self.root/'qualification.json'):'final',str(self.root/'c1-source-identity-proof-v4.json'):'proof',str(self.root/'prepared.json'):'prepared',str(Path(r.__file__)):'parent',str(self.root/'c1-post-model-identity-v4.json'):'identity',str(r.ROOT/'strata/flash-next/model-lock.json'):'lock',str(self.root/'post-health.json'):'health',str(self.root/'qualification-controller-v4.json'):'raw','mock-health':'healthlog','mock-collective':'collectivelog'}
  with patch.object(r.c1,'read',side_effect=lambda p:read[str(p)]),patch.object(r.c1,'sha',side_effect=lambda p:hashes[str(p)]),patch.object(r.c1,'stat_signature',return_value=self.signature):return r.validate_final_source_proof(self.root,self.prepared)
 def test_mock_complete_current_positive(self):self.assertTrue(self.validate()['passed'])
 def test_legacy_passed_controller_refused(self):
  self.final.pop('c1_parent_generation')
  with self.assertRaises(AssertionError):self.validate()
 def test_changed_preparation_engine_refused(self):
  self.prepared['engine_receipt_sha256']='old21'
  with self.assertRaises(AssertionError):self.validate()
 def test_missing_new_upload_postproof_refused(self):
  self.prepared['upload_lifecycle']['post_full4_source_qualified']=False
  with self.assertRaises(AssertionError):self.validate()
 def test_stale_prehash_before_health_refused(self):
  self.identity['started']=7.5
  with self.assertRaises(AssertionError):self.validate()
 def test_incomplete3_hash_refused(self):
  self.identity['rows'].pop()
  with self.assertRaises(AssertionError):self.validate()
 def test_wrong_publisher_bytes_refused(self):
  self.identity['rows'][2]['sha256']='bad'
  with self.assertRaises(AssertionError):self.validate()
 def test_changed_current_stat_refused(self):
  self.signature=[1,2,100,9,5]
  with self.assertRaises(AssertionError):self.validate()
 def test_nonfinite_chronology_refused(self):
  self.identity['started']=float('nan')
  with self.assertRaises(AssertionError):self.validate()
 def test_one_knownpage_only_refused(self):
  self.proof['known_pages_after_hash']['rows'].pop()
  with self.assertRaises(AssertionError):self.validate()
 def test_changed_knownpage_hash_refused(self):
  self.proof['known_pages_before_hash']['rows'][1]['sha256']='bad'
  with self.assertRaises(AssertionError):self.validate()
 def test_pageguards_not_bracketing_hash_refused(self):
  self.proof['known_pages_after_hash']['epoch']=10.
  with self.assertRaises(AssertionError):self.validate()
 def test_final_gate_requires_terminal_health_supervisor_source(self):
  result={'owned_terminal':True,'post_health_passed':True,'launch_supervisor_exit_code':0};self.assertTrue(r.finalizable(result,self.identity))
  for key in result:
   changed=dict(result);changed.pop(key);self.assertFalse(r.finalizable(changed,self.identity),key)
  changed=dict(result,error='sourceguardfail');self.assertFalse(r.finalizable(changed,self.identity))
 def test_parent_supervisor_receipt_helper_unchanged(self):
  old=ast.parse(Path(r.__file__).with_name('qualify_c1_serving.py').read_text());new=ast.parse(Path(r.__file__).read_text())
  find=lambda tree:next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='record_supervisor_exit')
  self.assertEqual(ast.dump(find(old)),ast.dump(find(new)))
 def test_pagefailure_exactviews_tiny_fixture(self):
  shards=[self.root/str(i) for i in range(4)];shards[2].write_bytes(b'a'*4096+b'b'*4096)
  with patch.object(r.watchdog,'KNOWN_PAGES',((0,'0'*64),(4096,'1'*64))):
   with self.assertRaises(ValueError):r.upload_v2.guarded_pages(shards,self.root,'CPU_mock')
  evidence=json.loads(next(self.root.glob('*source-pages.json')).read_text());self.assertEqual([Path(row['preserved_path']).read_bytes() for row in evidence['rows']],[b'a'*4096,b'b'*4096])
 def test_missing_or_failed_parent_refused(self):
  self.parent['passed']=False
  with self.assertRaises(AssertionError):self.validate()
 def test_parent_final_proof_crossbind_refused(self):
  self.parent['qualification_sha256']='old'
  with self.assertRaises(AssertionError):self.validate()
 def test_pinned_source_modules(self):r.source_pins()

if __name__=='__main__':unittest.main()
