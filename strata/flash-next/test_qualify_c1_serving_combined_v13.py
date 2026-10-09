#!/usr/bin/env python3
"""CPU metadata/synthetic controls only; no real model/GPU/qualification."""
import ast,copy,json,tempfile,time,unittest,os
import test_c1_serve_controller_combined_v13 as c1_fixture
from pathlib import Path
from unittest.mock import patch
import qualify_c1_serving_combined_v13 as r

class FinalTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.signature=[1,2,100,4,5]
  self.lock={'revision':'CPU_mock','destination':'models/mock','files':[{'path':'UD-Q4_K_XL/mock-0000%d-of-00004.gguf'%i,'sha256':str(i)*64,'size':100} for i in range(1,5)]}
  rows=[{'path':str(r.ROOT/self.lock['destination']/f['path']),'passed':True,'sha256':f['sha256'],'expected_sha256':f['sha256'],'bytes':100,'stat_before':self.signature,'stat_after':self.signature} for f in self.lock['files']]
  self.identity={'passed':True,'rows':rows,'started':9.1,'finished':12.2,'after_terminal_and_post_health_epoch':8.2,'lock_sha256':'lock','model_revision':'CPU_mock'}
  third=next(row['path'] for row in rows if '00003-of-00004' in row['path'])
  def guard(epoch):return {'passed':True,'path':third,'stat_before':self.signature,'stat_after':self.signature,'epoch':epoch,'rows':[{'offset':offset,'bytes':4096,'sha256':sha,'expected_sha256':sha,'passed':True} for offset,sha in r.watchdog.KNOWN_PAGES]}
  self.proof={'passed':True,'parent_controller_sha256':'parent','controller_sha256':r.CONTROLLER_SHA,'watchdog_sha256':r.WATCHDOG_SHA,'prepared_sha256':'prepared','engine_receipt_sha256':'engine','upload_v2_post_identity_sha256':'uploadpost','terminal_epoch':7.2,'post_health_finished_epoch':8.2,'post_health_sha256':'health','identity':{'path':str(self.root/'c1-post-model-identity-v13.json'),'sha256':'identity'},'known_pages_before_hash':guard(8.9),'known_pages_after_hash':guard(12.3),'page_guard_calls':3,'between_polls_identity_observed':False}
  self.final={'passed':True,'teardown_passed':True,'post_health_passed':True,'c1_parent_generation':13,'post_full4_source_qualified':True,'engine_receipt_sha256':'engine','post_health_sha256':'health','c1_parent_controller_sha256':'parent','controller_qualification_sha256':'raw','c1_source_identity_proof':{'path':str(self.root/'c1-source-identity-proof-v13.json'),'sha256':'proof'}}
  self.prepared={'combined_generation':{'prompt_verifier_P30_capture35':True,'final_window_PLE_observer_fix34':True,'current_PLE_host_observer_source33':True,'semantic_current_PLE_gather_fix32':True,'plan_sha256':r.c1.COMBINED_PLAN_SHA,'source_count':63,'header_payloads':[{}]*27,'sdk_targets':['mock']*8,'runtime_python_sources':r.c1.PYTHON_SOURCE_SHA},'engine_receipt_sha256':'engine','upload_lifecycle':{'post_model_identity_sha256':'uploadpost','post_full4_source_qualified':True}}
  self.proof['combined_generation']=self.prepared['combined_generation']
  self.parent={'passed':True,'parent_generation':13,'parent_controller_sha256':'parent','owned_terminal':True,'post_health_passed':True,'launch_supervisor_exit_code':0,'qualification_sha256':'final','source_identity_proof_sha256':'proof','engine_receipt_sha256':'engine','post_model_identity':{'sha256':'identity','passed':True},'owned_terminal_epoch':7.2,'post_health_finished_epoch':8.2,'finished_epoch':13.}
  self.health={'passed':True,'finished_epoch':8.2,'files':[{'path':'mock-health','sha256':'healthlog'},{'path':'mock-collective','sha256':'collectivelog'}]}
 def validate(self):
  read={str(self.root/'parent-qualification.json'):self.parent,str(self.root/'qualification.json'):self.final,str(self.root/'c1-source-identity-proof-v13.json'):self.proof,str(self.root/'c1-post-model-identity-v13.json'):self.identity,str(self.root/'post-health.json'):self.health,str(r.ROOT/'strata/flash-next/model-lock.json'):self.lock}
  hashes={str(self.root/'qualification.json'):'final',str(self.root/'c1-source-identity-proof-v13.json'):'proof',str(self.root/'prepared.json'):'prepared',str(Path(r.__file__)):'parent',str(self.root/'c1-post-model-identity-v13.json'):'identity',str(r.ROOT/'strata/flash-next/model-lock.json'):'lock',str(self.root/'post-health.json'):'health',str(self.root/'qualification-controller-v13.json'):'raw','mock-health':'healthlog','mock-collective':'collectivelog'}
  with patch.object(r.c1,'read',side_effect=lambda p:read[str(p)]),patch.object(r.c1,'sha',side_effect=lambda p:hashes[str(p)]),patch.object(r.c1,'stat_signature',return_value=self.signature):return r.validate_final_source_proof(self.root,self.prepared)
 def test_parent10_proof_refused(self):
  self.final['c1_parent_generation']=10
  with self.assertRaises(AssertionError):self.validate()
 def test_C112_parent_proof_refused(self):
  self.final['c1_parent_generation']=12
  with self.assertRaises(AssertionError):self.validate()
 def test_missing_source35_binding_refused(self):
  self.prepared['combined_generation'].pop('prompt_verifier_P30_capture35')
  with self.assertRaises(AssertionError):self.validate()
 def test_C111_parent_proof_refused(self):
  self.final['c1_parent_generation']=11
  with self.assertRaises(AssertionError):self.validate()
 def test_source34_binding_missing_refused(self):
  self.prepared['combined_generation'].pop('final_window_PLE_observer_fix34')
  with self.assertRaises(AssertionError):self.validate()
 def test_source34_binding_false_refused(self):
  self.prepared['combined_generation']['final_window_PLE_observer_fix34']=False
  with self.assertRaises(AssertionError):self.validate()
 def test_source33_binding_missing_refused(self):
  self.prepared['combined_generation'].pop('current_PLE_host_observer_source33')
  with self.assertRaises(AssertionError):self.validate()
 def test_missing_semantic_fix32_binding_rejected(self):
  self.prepared['combined_generation'].pop('semantic_current_PLE_gather_fix32')
  with self.assertRaises(AssertionError):self.validate()
 def test_prior_generation9_final_proof_rejected(self):
  self.final['c1_parent_generation']=9
  with self.assertRaises(AssertionError):self.validate()
 def test_old_generation_preparation_rejected(self):
  self.prepared['combined_generation']['plan_sha256']='old28'
  with self.assertRaises(AssertionError):self.validate()
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


class ParentActualProducerConsumerTests(unittest.TestCase):
 def setUp(self):
  self.fixture=c1_fixture.ActualStatProducerConsumerTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);f=self.fixture;c=r.c1;self.directory=f.out;self.rows=f.identity['rows']
  # Real tiny-file identity/stat/page records; only API/GPU lifecycle flags mocked.
  self.prepared={'combined_generation':{'prompt_verifier_P30_capture35':True,'final_window_PLE_observer_fix34':True,'current_PLE_host_observer_source33':True,'semantic_current_PLE_gather_fix32':True,'plan_sha256':c.COMBINED_PLAN_SHA,'source_count':63,'header_payloads':[{}]*27,'sdk_targets':['mock']*8,'runtime_python_sources':c.PYTHON_SOURCE_SHA},'scope':'CPU mock preparation only','engine_receipt_sha256':c.sha(f.enginepath),'upload_lifecycle':{'post_model_identity_sha256':c.sha(f.identitypath),'post_full4_source_qualified':True}};c.write(self.directory/'prepared.json',self.prepared)
  self.path=self.directory/'c1-post-model-identity-v13.json';c.write(self.path,f.identity)
  health_files=[]
  for name in ('CPU-mock-health.log','CPU-mock-collective.log'):
   path=self.directory/name;path.write_text('CPU mock only; not GPU health evidence\n');health_files.append({'path':str(path),'sha256':c.sha(path)})
  health={'scope':'CPU mock lifecycle metadata','passed':True,'finished_epoch':f.report['post_health_finished_epoch'],'files':health_files};c.write(self.directory/'post-health.json',health)
  raw={'scope':'CPU mock controller result only','passed':True};c.write(self.directory/'qualification-controller-v13.json',raw)
  self.proof={'combined_generation':self.prepared['combined_generation'],'passed':True,'parent_controller_sha256':c.sha(Path(r.__file__)),'controller_sha256':r.CONTROLLER_SHA,'watchdog_sha256':r.WATCHDOG_SHA,'prepared_sha256':c.sha(self.directory/'prepared.json'),'engine_receipt_sha256':self.prepared['engine_receipt_sha256'],'upload_v2_post_identity_sha256':c.sha(f.identitypath),'terminal_epoch':f.report['owned_terminal_epoch'],'post_health_finished_epoch':f.report['post_health_finished_epoch'],'post_health_sha256':c.sha(self.directory/'post-health.json'),'identity':{'path':str(self.path),'sha256':c.sha(self.path)},'known_pages_before_hash':f.report['known_pages_before_post_hash'],'known_pages_after_hash':f.report['known_pages_after_post_hash'],'page_guard_calls':3,'between_polls_identity_observed':False}
  proofpath=self.directory/'c1-source-identity-proof-v13.json';c.write(proofpath,self.proof)
  final={'scope':'CPU mock boundary only','passed':True,'teardown_passed':True,'post_health_passed':True,'c1_parent_generation':13,'post_full4_source_qualified':True,'engine_receipt_sha256':self.prepared['engine_receipt_sha256'],'post_health_sha256':self.proof['post_health_sha256'],'c1_parent_controller_sha256':self.proof['parent_controller_sha256'],'controller_qualification_sha256':c.sha(self.directory/'qualification-controller-v13.json'),'c1_source_identity_proof':{'path':str(proofpath),'sha256':c.sha(proofpath)}};c.write(self.directory/'qualification.json',final)
  parent={'scope':'CPU mock parent only','passed':True,'parent_generation':13,'parent_controller_sha256':self.proof['parent_controller_sha256'],'owned_terminal':True,'post_health_passed':True,'launch_supervisor_exit_code':0,'qualification_sha256':c.sha(self.directory/'qualification.json'),'source_identity_proof_sha256':c.sha(proofpath),'engine_receipt_sha256':self.prepared['engine_receipt_sha256'],'post_model_identity':{'sha256':c.sha(self.path),'passed':True},'owned_terminal_epoch':self.proof['terminal_epoch'],'post_health_finished_epoch':self.proof['post_health_finished_epoch'],'finished_epoch':time.time()};c.write(self.directory/'parent-qualification.json',parent)
 def consume(self):
  # Actual os.stat/hash/read; no monkeypatched list/dict stat return.
  with patch.object(r,'ROOT',self.fixture.root),patch.object(r.watchdog,'KNOWN_PAGES',self.fixture.pages):return r.validate_final_source_proof(self.directory,self.prepared)
 def test_actual_four_tiny_files_parent_consumer_positive(self):self.assertTrue(self.consume()['passed'])
 def test_actual_ctime_change_parent_refused(self):
  path=self.fixture.shards[2];before=path.stat();mode=before.st_mode&0o777;os.chmod(path,mode^0o100);os.chmod(path,mode);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
  self.assertEqual(before.st_mtime_ns,path.stat().st_mtime_ns);self.assertNotEqual(before.st_ctime_ns,path.stat().st_ctime_ns)
  with self.assertRaises(AssertionError):self.consume()

if __name__=='__main__':unittest.main()
