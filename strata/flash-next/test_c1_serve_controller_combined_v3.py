#!/usr/bin/env python3
"""Mock CPU admission controls. No real engine/upload/model result is created."""
import ast,copy,json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import c1_serve_controller_combined_v3 as c

class GenerationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.engine=Path(self.temp.name).resolve()
  self.plan=c.read(c.COMBINED_PLAN);self.receipt={'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'image':c.BASE_IMAGE,'plan_sha256':c.COMBINED_PLAN_SHA,'plan_snapshot':str(self.engine/'plan.snapshot.json'),'source_revision':self.plan['source_revision'],'ggml_revision':self.plan['ggml']['revision'],'patches':self.plan['patches'],'source_copy':str(self.engine/'source'),'patched_source_sha256':dict(self.plan['expected_patched_source_sha256']),'binary_sha256':{str(self.engine/'build'/target):'b'*64 for target in self.plan['build_targets']}}
  self.hashes={str(c.COMBINED_PLAN):c.COMBINED_PLAN_SHA,str(self.engine/'plan.snapshot.json'):c.COMBINED_PLAN_SHA,str(self.engine/'receipt.json'):'c'*64}
  self.hashes.update({str(self.engine/'source'/name):value for name,value in self.receipt['patched_source_sha256'].items()});self.hashes.update({str(self.engine/'source'/name):value for name,value in c.PYTHON_SOURCE_SHA.items()});self.hashes.update(self.receipt['binary_sha256'])
 def gate(self):
  def read(path):
   if Path(path)==c.COMBINED_PLAN:return self.plan
   if Path(path)==self.engine/'receipt.json':return self.receipt
   raise AssertionError('Unexpected mock read '+str(path))
  with patch.object(c,'read',side_effect=read),patch.object(c,'sha',side_effect=lambda path:self.hashes[str(path)]):return c.combined_generation_gate(self.engine)
 def test_mock_complete_generation_positive(self):
  result=self.gate();self.assertEqual(result['source_count'],51);self.assertEqual(len(result['sdk_targets']),8);self.assertEqual(len(result['runtime_python_sources']),6);self.assertFalse(result['full_model_math_qualified'])
 def test_missing_failed_sdk_rejected(self):
  self.receipt['build_rc']=None
  with self.assertRaises(ValueError):self.gate()
 def test_old_engine_plan_rejected(self):
  self.receipt['plan_sha256']='old20'
  with self.assertRaises(ValueError):self.gate()
 def test_incomplete_source_ledger_rejected(self):
  self.receipt['patched_source_sha256'].pop(next(iter(self.receipt['patched_source_sha256'])))
  with self.assertRaises(ValueError):self.gate()
 def test_actual_source_byte_change_rejected(self):
  name=next(iter(self.receipt['patched_source_sha256']));self.hashes[str(self.engine/'source'/name)]='0'*64
  with self.assertRaises(ValueError):self.gate()
 def test_missing_sdk_target_rejected(self):
  self.receipt['binary_sha256'].pop(next(iter(self.receipt['binary_sha256'])))
  with self.assertRaises(ValueError):self.gate()
 def test_binary_changed_rejected(self):
  self.hashes[next(iter(self.receipt['binary_sha256']))]='0'*64
  with self.assertRaises(ValueError):self.gate()
 def test_sixth_runtime_source_changed_rejected(self):
  self.hashes[str(self.engine/'source'/'serve/batch_request_identity.py')]='0'*64
  with self.assertRaises(ValueError):self.gate()
 def test_prepare_missing_genuine_prereqs_stops_before_hash(self):
  args=SimpleNamespace(verify_model_shards=False,runtime_receipt=None,upload_lifecycle=None,oracle_receipt=None)
  with patch.object(c,'sha',side_effect=AssertionError('No actual payload read authorized')):
   with self.assertRaises(ValueError):c.prepare(args)
 def test_source_runtime_primary_and_alias_topologies(self):
  self.assertEqual(c.PRIMARY,'hotschmoe-dd');self.assertEqual(len(c.PYTHON_SOURCES),6)
  for profile in c.PROFILES.values():
   self.assertIn('Unsloth-UD-Q4_K_XL-strata-native-source-hc-ple-mtp0',profile['alias']);self.assertIn('ctx'+str(profile['context']),profile['alias']);self.assertTrue(profile['alias'].endswith('gpu0' if profile['cards']==[0] else 'gpu0-1-split32-16'))
 def test_preserved_lifecycle_helpers(self):
  baseline=ast.parse(Path(c.__file__).with_name('c1_serve_controller.py').read_text());new=ast.parse(Path(c.__file__).read_text());functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
  old=functions(baseline);current=functions(new)
  for name in ('upload_gate','full_source_case_gate','segmented_profile_gate','leased','inspected','absent','owned_stop','poll_owned_run','screen'):
   self.assertEqual(old[name],current[name],name)
 def test_two_page_guard_spec(self):
  self.assertEqual([offset for offset,_ in c.KNOWN_SOURCE_PAGES],[3857879040,39437303808]);self.assertTrue(all(len(digest)==64 for _,digest in c.KNOWN_SOURCE_PAGES))

 def test_two_synthetic_pages_positive_and_second_corruption(self):
  path=self.engine/'fixture-00003-of-00004.gguf';a=b'a'*4096;b=b'b'*4096;path.write_bytes(a+b)
  pages=((0,hashlib.sha256(a).hexdigest()),(4096,hashlib.sha256(b).hexdigest()))
  with patch.object(c,'KNOWN_SOURCE_PAGES',pages):
   result=c.original_page_sentinel([{'path':str(path)}]);self.assertEqual(len(result['pages']),2)
   path.write_bytes(a+b'c'*4096)
   with self.assertRaises(ValueError):c.original_page_sentinel([{'path':str(path)}])
 def test_old_five_manifest_rejected_before_payload(self):
  m={'combined_generation':{'mock':True},'engine_receipt':str(self.engine/'receipt.json'),'runtime_python_source_count':6}
  manifest={'runtime':{'python_sources':{k:v for k,v in c.PYTHON_SOURCE_SHA.items() if k!='serve/batch_request_identity.py'}}}
  with patch.object(c,'combined_generation_gate',return_value={'mock':True}),patch.object(c,'read',side_effect=[m,manifest]),patch.object(c,'sha',side_effect=AssertionError('No payload reads before manifest rejection')):
   with self.assertRaises(ValueError):c.validate_prepared(self.engine)

if __name__=='__main__':unittest.main()
