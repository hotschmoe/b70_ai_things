#!/usr/bin/env python3
"""Mock CPU admission controls. No real engine/upload/model result is created."""
import ast,copy,json,tempfile,unittest,hashlib,time,os,contextlib,io
import run_source_upload_oracle_full_v2 as producer
import source_page_watchdog_v3 as watchdog
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import c1_serve_controller_combined_v8 as c

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
  result=self.gate();self.assertEqual(result['source_count'],60);self.assertEqual(len(result['sdk_targets']),8);self.assertEqual(len(result['runtime_python_sources']),6);self.assertFalse(result['full_model_math_qualified'])
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
 def test_prior_server_import_source_rejected(self):
  self.hashes[str(self.engine/'source'/'serve/server.py')]='b11880b1abeba42a57bd7dbf87780eace5e6446567c4ec1833b17d2529fd24de'
  with self.assertRaises(ValueError):self.gate()
 def test_integrated_recipe_exact_closure(self):
  plan=c.read(c.COMBINED_PLAN)
  self.assertEqual(len(plan['added_header_payloads']),24)
  self.assertEqual(len(plan['expected_patched_source_sha256']),60)
  self.assertEqual(plan['runtime_python_sources'],c.PYTHON_SOURCE_SHA)
  self.assertTrue(all('integrated25-29-' in p['alias'] for p in c.PROFILES.values()))
 def test_missing_header_payload_rejected(self):
  self.plan['added_header_payloads'].pop()
  with self.assertRaises(ValueError):self.gate()
 def test_baseline_activation_rejected(self):
  for name in c.BASELINE_OFF_FLAGS:
   with self.assertRaises(ValueError):c.baseline_profile_gate(['--batch','0'],{name:'1'})
  c.baseline_profile_gate(['--batch','0'],{})
 def test_wrong_header_digest_rejected(self):
  self.plan['added_header_payloads'][0]['sha256']='0'*64
  with self.assertRaises(ValueError):self.gate()
 def test_metadata_missing_new_sdk_never_reaches_upload_or_page(self):
  prepared={'engine_receipt':str(self.engine/'receipt.json')}
  with patch.object(c,'combined_generation_gate',side_effect=ValueError('new SDK missing')),patch.object(c,'upload_gate',side_effect=AssertionError('No upload access')),patch.object(c,'original_page_sentinel',side_effect=AssertionError('No page read')):
   with self.assertRaises(ValueError):c.metadata_admission_gate(prepared)
 def test_metadata_prior_prepared_rejected_before_upload(self):
  prepared={'engine_receipt':str(self.engine/'receipt.json'),'combined_generation':{'plan_sha256':'old28'},'launch_allowed':True}
  with patch.object(c,'combined_generation_gate',return_value={'plan_sha256':c.COMBINED_PLAN_SHA}),patch.object(c,'upload_gate',side_effect=AssertionError('No upload access')):
   with self.assertRaises(ValueError):c.metadata_admission_gate(prepared)
 def test_metadata_mock_integrated_runtime_positive_and_negative(self):
  generation={'scope':'CPU mock new SDK only','engine_receipt_sha256':'mockengine'}
  prepared={'engine_receipt':str(self.engine/'receipt.json'),'combined_generation':generation,'launch_allowed':True,'controller_sha256':'mockcontroller','engine_receipt_sha256':'mockengine','upload_lifecycle':{'path':'mockupload','oracle':'mockoracle'},'pack_receipt':'mockpack','runtime_receipt':'mockruntime','runtime_receipt_sha256':'mockruntimehash'}
  runtime={'passed':True,'base_image':c.BASE_IMAGE,'gpu_libraries_unchanged':True,'packages_path':'mockpackages','packages_sha256':'mockpackageshash'}
  def digest(path):return {'mockruntime':'mockruntimehash','mockpackages':'mockpackageshash'}.get(str(path),'mockcontroller')
  with patch.object(c,'combined_generation_gate',return_value=generation),patch.object(c,'upload_gate',return_value={'scope':'CPU mock upload only'}),patch.object(c,'read',return_value=runtime),patch.object(c,'sha',side_effect=digest),patch.object(c,'original_page_sentinel',side_effect=AssertionError('metadata must not read model payload')):
   self.assertEqual(c.metadata_admission_gate(prepared),generation)
   runtime['gpu_libraries_unchanged']=False
   with self.assertRaises(ValueError):c.metadata_admission_gate(prepared)
 def test_parent_admission_precedes_payload_probe(self):
  source=Path(c.__file__).with_name('qualify_c1_serving_combined_v8.py').read_text()
  self.assertLess(source.index('c1.metadata_admission_gate(raw_prepared)'),source.index("upload_v2.guarded_pages(shards, out, 'c1-before-prepared-validation')"))
 def test_missing_owner29_patch_rejected(self):
  self.receipt['patches']=self.receipt['patches'][:-1]
  with self.assertRaises(ValueError):self.gate()
 def test_prior28_arena_source_rejected(self):
  name='sycl/include/strata/core/slot_session_arena.hpp'
  old=c.read(Path(c.__file__).with_name('integrated-batch-public-prefix-engine-build-plan-v1.json'))
  self.assertNotEqual(old['expected_patched_source_sha256'][name],self.plan['expected_patched_source_sha256'][name])
  self.hashes[str(self.engine/'source'/name)]=old['expected_patched_source_sha256'][name]
  with self.assertRaises(ValueError):self.gate()
 def test_real_cpu_package_metadata_positive_and_mutation(self):
  packages=self.engine/'cpu-packages.json';packages.write_text('CPU fixture packages only\n')
  runtime_path=self.engine/'cpu-runtime.json'
  c.write(runtime_path,{'passed':True,'base_image':c.BASE_IMAGE,'gpu_libraries_unchanged':True,'packages_path':str(packages),'packages_sha256':c.sha(packages)})
  generation={'scope':'CPU mocked SDK29; no compiled qualification','engine_receipt_sha256':'cpu_mock_engine'}
  prepared={'engine_receipt':str(self.engine/'receipt.json'),'combined_generation':generation,'launch_allowed':True,'controller_sha256':c.sha(Path(c.__file__)),'engine_receipt_sha256':'cpu_mock_engine','upload_lifecycle':{'path':'CPU mock upload','oracle':'CPU mock oracle'},'pack_receipt':'CPU mock intake','runtime_receipt':str(runtime_path),'runtime_receipt_sha256':c.sha(runtime_path)}
  with patch.object(c,'combined_generation_gate',return_value=generation),patch.object(c,'upload_gate',return_value={'scope':'CPU mocked upload only'}),patch.object(c,'original_page_sentinel',side_effect=AssertionError('No model page read')):
   self.assertEqual(c.metadata_admission_gate(prepared),generation)
   packages.write_text('CPU fixture changed package bytes\n')
   with self.assertRaises(ValueError):c.metadata_admission_gate(prepared)
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
  for name in ('full_source_case_gate','segmented_profile_gate','leased','inspected','absent','owned_stop','poll_owned_run','screen'):
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


class StrictUploadTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.path=self.root/'receipt.json';self.oraclepath=self.root/'oracle.json';self.enginepath=self.root/'engine.json';self.identitypath=self.root/'post-model-identity.json'
  self.signature=[1,2,100,4,5];self.lock={'revision':'CPU_mock','destination':'models/mock','files':[{'path':'UD-Q4_K_XL/mock-0000%d-of-00004.gguf'%i,'sha256':str(i)*64,'size':100} for i in range(1,5)]}
  files={str(c.REPO/self.lock['destination']/f['path']):f for f in self.lock['files']};self.third=next(name for name in files if '00003-of-00004' in name)
  rows=[{'path':name,'passed':True,'sha256':f['sha256'],'expected_sha256':f['sha256'],'bytes':100,'stat_before':self.signature,'stat_after':self.signature} for name,f in files.items()]
  self.identity={'passed':True,'rows':rows,'started':9.,'finished':12.,'after_terminal_and_post_health_epoch':8.,'lock_sha256':'lock','model_revision':'CPU_mock','read_mode':'ordinary complete buffered read; no direct IO/cache mutation'}
  self.report={'passed':True,'runner_generation':2,'controller_sha256':c.UPLOAD_RUNNER_V2_SHA,'watchdog_sha256':c.UPLOAD_WATCHDOG_SHA,'engine_receipt':str(self.enginepath),'engine_receipt_sha256':'engine','oracle_receipt_sha256':'oracle','owned_terminal':True,'post_health_passed':True,'post_identity_complete4':True,'owned_terminal_epoch':7.,'post_health_finished_epoch':8.,'finished_epoch':14.,'started_epoch':1.,'post_model_identity':{'passed':True,'path':str(self.identitypath),'sha256':'identity','started':9.,'finished':12.},'source_guard_between_polls_unobserved':True,'source_guard_poll_seconds':2,'expected_cases':['case%d'%i for i in range(5)],'cases':[{} for _ in range(5)]}
  def guard(epoch):return {'passed':True,'path':self.third,'stat_before':self.signature,'stat_after':self.signature,'epoch':epoch,'rows':[{'offset':offset,'bytes':4096,'sha256':sha,'expected_sha256':sha,'passed':True} for offset,sha in c.KNOWN_SOURCE_PAGES]}
  self.report.update(known_pages_admission=guard(2.),known_pages_before_post_hash=guard(8.5),known_pages_after_post_hash=guard(13.))
  for case in self.report['expected_cases']:self.report['known_pages_before_'+case]=guard(3.);self.report['known_pages_after_'+case]=guard(4.)
  self.oracle={'engine_receipt':str(self.enginepath),'engine_receipt_sha256':'engine'}
 def strict(self):
  reads={str(self.path):self.report,str(self.oraclepath):self.oracle,str(self.identitypath):self.identity,str(c.REPO/'strata/flash-next/model-lock.json'):self.lock}
  hashes={str(self.enginepath):'engine',str(self.oraclepath):'oracle',str(self.identitypath):'identity',str(c.REPO/'strata/flash-next/model-lock.json'):'lock',str(c.REPO/'strata/flash-next/run_source_upload_oracle_full_v2.py'):c.UPLOAD_RUNNER_V2_SHA,str(c.REPO/'strata/flash-next/source_page_watchdog_v3.py'):c.UPLOAD_WATCHDOG_SHA,str(self.root/'controller.py'):c.UPLOAD_RUNNER_V2_SHA,str(self.root/'source-page-watchdog.py'):c.UPLOAD_WATCHDOG_SHA}
  with patch.object(c,'read',side_effect=lambda p:reads[str(p)]),patch.object(c,'sha',side_effect=lambda p:hashes[str(p)]),patch.object(c,'stat_signature',return_value=self.signature):return c.strict_v2_upload_provenance(self.path,self.oraclepath,self.enginepath)
 def test_mock_v2_complete_provenance(self):self.assertTrue(self.strict()['post_full4_source_qualified'])
 def test_fractional_finish_ordering_positive(self):
  self.identity['finished']=12.9;self.report['post_model_identity']['finished']=12.9;self.report['known_pages_after_post_hash']['epoch']=12.905;self.report['finished_epoch']=12.91
  self.assertTrue(self.strict()['post_full4_source_qualified'])
 def test_legacy_passed_upload_new_sdk_rejected(self):
  self.report.pop('runner_generation');self.report['passed']=True
  with self.assertRaises(ValueError):self.strict()
 def test_missing_postfull4_rejected(self):
  self.report.pop('post_model_identity')
  with self.assertRaises(ValueError):self.strict()
 def test_old_oracle_engine_crossbind_rejected(self):
  self.oracle['engine_receipt_sha256']='old21'
  with self.assertRaises(ValueError):self.strict()
 def test_hash_before_health_rejected(self):
  self.identity['started']=7.5;self.report['post_model_identity']['started']=7.5
  with self.assertRaises(ValueError):self.strict()
 def test_hash_finishes_after_pass_rejected(self):
  self.report['finished_epoch']=11.
  with self.assertRaises(ValueError):self.strict()
 def test_corrupt_source_identity_sha_rejected(self):
  self.identity['rows'][2]['sha256']='bad'
  with self.assertRaises(ValueError):self.strict()
 def test_incomplete3_identity_rejected(self):
  self.identity['rows'].pop()
  with self.assertRaises(ValueError):self.strict()
 def test_one_page_only_guard_rejected(self):
  self.report['known_pages_admission']['rows'].pop()
  with self.assertRaises(ValueError):self.strict()
 def test_post_hash_guard_before_hash_end_rejected(self):
  self.report['known_pages_after_post_hash']['epoch']=11.
  with self.assertRaises(ValueError):self.strict()
 def test_changed_runner_or_watchdog_rejected(self):
  self.report['watchdog_sha256']='bad'
  with self.assertRaises(ValueError):self.strict()
 def test_original_sourcecase_gate_body_preserved(self):
  tree=ast.parse(Path(c.__file__).read_text());oldtree=ast.parse(Path(c.__file__).with_name('c1_serve_controller_combined_v3.py').read_text())
  new=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='original_upload_gate');old=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='upload_gate')
  self.assertEqual([ast.dump(n) for n in new.body],[ast.dump(n) for n in old.body])


class ActualStatProducerConsumerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  repo=c.REPO;tools=self.root/'strata/flash-next';tools.mkdir(parents=True);self.out=self.root/'CPU-mock-upload';self.out.mkdir()
  for name in ('run_source_upload_oracle_full_v2.py','source_page_watchdog_v3.py'):(tools/name).write_bytes((repo/'strata/flash-next'/name).read_bytes())
  (self.out/'controller.py').write_bytes((tools/'run_source_upload_oracle_full_v2.py').read_bytes());(self.out/'source-page-watchdog.py').write_bytes((tools/'source_page_watchdog_v3.py').read_bytes())
  self.lock={'revision':'CPU_mock_not_actual_model','destination':'models/mock','files':[]};self.shards=[]
  for i in range(1,5):
   rel='UD-Q4_K_XL/mock-0000%d-of-00004.gguf'%i;path=self.root/self.lock['destination']/rel;path.parent.mkdir(parents=True,exist_ok=True);raw=bytes([i])*8192;path.write_bytes(raw);self.shards.append(path);self.lock['files'].append({'path':rel,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  self.lockpath=tools/'model-lock.json';c.write(self.lockpath,self.lock)
  raw=self.shards[2].read_bytes();self.pages=((0,hashlib.sha256(raw[:4096]).hexdigest()),(4096,hashlib.sha256(raw[4096:]).hexdigest()))
  def guard():
   value=watchdog.inspect_pages(self.shards[2],self.pages)
   for row in value['rows']:row.pop('data')
   return value
  before=guard();after=guard();terminal=time.time();health=time.time();beforehash=guard()
  self.identitypath=self.out/'post-model-identity.json';self.identity=producer.full_buffered_identity(self.lockpath,self.lock,self.shards,self.identitypath,max(terminal,health));afterhash=guard()
  self.enginepath=self.root/'CPU-mock-engine.json';c.write(self.enginepath,{'scope':'CPU mock engine only'})
  self.oraclepath=self.root/'CPU-mock-oracle.json';c.write(self.oraclepath,{'engine_receipt':str(self.enginepath.resolve()),'engine_receipt_sha256':c.sha(self.enginepath)})
  self.report={'scope':'CPU mocked GPU flags/cases; REAL tiny-file producer-consumer source-schema test only','passed':True,'runner_generation':2,'controller_sha256':c.UPLOAD_RUNNER_V2_SHA,'watchdog_sha256':c.UPLOAD_WATCHDOG_SHA,'engine_receipt':str(self.enginepath.resolve()),'engine_receipt_sha256':c.sha(self.enginepath),'oracle_receipt_sha256':c.sha(self.oraclepath),'owned_terminal':True,'post_health_passed':True,'post_identity_complete4':True,'owned_terminal_epoch':terminal,'post_health_finished_epoch':health,'finished_epoch':time.time(),'started_epoch':before['epoch']-.01,'post_model_identity':{'passed':True,'path':str(self.identitypath),'sha256':c.sha(self.identitypath),'started':self.identity['started'],'finished':self.identity['finished']},'source_guard_between_polls_unobserved':True,'source_guard_poll_seconds':2,'expected_cases':['CPU-mock-case%d'%i for i in range(5)],'cases':[{} for _ in range(5)],'known_pages_admission':before,'known_pages_before_post_hash':beforehash,'known_pages_after_post_hash':afterhash}
  for case in self.report['expected_cases']:self.report['known_pages_before_'+case]=before;self.report['known_pages_after_'+case]=after
  self.reportpath=self.out/'receipt.json';c.write(self.reportpath,self.report)
 def consume(self):
  # NO stat/read/sha mocks: actual producer lists consumed against actual os.stat.
  with patch.object(c,'REPO',self.root),patch.object(c,'KNOWN_SOURCE_PAGES',self.pages):return c.strict_v2_upload_provenance(self.reportpath,self.oraclepath,self.enginepath)
 def test_prepare_actual_tinyfile_model_stat_and_page_metadata(self):
  # Full prepare body with explicitly mocked SDK/upload/runtime admission,
  # real tiny source files/hashes/stat/page guards; not a genuine preparation.
  tools=self.root/'strata/flash-next'
  for name in ('c1_trace_contract.py','c1_api_trace.py'):(tools/name).write_bytes((c.REPO/'strata/flash-next'/name).read_bytes())
  pack=self.root/'CPU-mock-pack';(pack/'tokenizer').mkdir(parents=True)
  for name in c.FILES:(pack/'tokenizer'/name).write_text('CPU mock tokenizer only\n')
  intake=self.root/'CPU-mock-intake.json';c.write(intake,{'metadata_complete':True,'RESULT':{'unexpected_inexact_conversions':[],'files':{}}})
  packages=self.root/'CPU-mock-packages.json';packages.write_text('CPU mock packages only\n')
  runtime=self.root/'CPU-mock-runtime.json';c.write(runtime,{'scope':'CPU mock admission only','passed':True,'base_image':c.BASE_IMAGE,'gpu_libraries_unchanged':True,'packages_path':str(packages),'packages_sha256':c.sha(packages),'inherited_math_env':{},'python_versions':{}})
  engine=self.root/'CPU-mock-preparation-engine';(engine/'build').mkdir(parents=True);exe=engine/'build/strata';exe.write_text('CPU mock executable; never run\n');(engine/'source/data').mkdir(parents=True);(engine/'source/data/expert-profile.bin').write_bytes(b'CPU mock profile')
  for name in c.PYTHON_SOURCES:
   path=engine/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU mock Python source; never imported\n')
  c.write(engine/'receipt.json',{'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'image':c.BASE_IMAGE,'binary_sha256':{str(exe):c.sha(exe)},'patched_source_sha256':{}})
  registry=self.root/'evals/configs/models.yaml';registry.parent.mkdir(parents=True);registry.write_text('served_model_id: '+c.PROFILES['one-card']['alias']+'\n')
  launch=c.read(c.REPO/'strata/flash-next/native-hc-launch-plan.json');c.write(tools/'native-hc-launch-plan.json',launch)
  self.lock['repo']='CPU_mock';c.write(self.lockpath,self.lock)
  output=self.root/'CPU-mock-prepared';args=SimpleNamespace(output=output,profile='one-card',engine_root=engine,runtime_receipt=runtime,upload_lifecycle=self.reportpath,oracle_receipt=self.oraclepath,verify_model_shards=True,port=18082)
  with patch.object(c,'REPO',self.root),patch.object(c,'PACK',pack),patch.object(c,'INTAKE',intake),patch.object(c,'KNOWN_SOURCE_PAGES',self.pages),patch.object(c,'combined_generation_gate',return_value={'scope':'CPU mock SDK only'}),patch.object(c,'upload_gate',return_value={'scope':'CPU mock GPU upload only'}),contextlib.redirect_stdout(io.StringIO()):c.prepare(args)
  model=c.read(output/'prepared.json');self.assertEqual(len(model['model_shards']),4)
  for row,path in zip(model['model_shards'],self.shards):self.assertEqual(row['stat'],producer.watchdog.signature(path));self.assertEqual(len(row['stat']),5);self.assertEqual(row['verified_sha256'],row['sha256'])
  with patch.object(c,'KNOWN_SOURCE_PAGES',self.pages):self.assertEqual(model['source_page_sentinel'],c.original_page_sentinel(model['model_shards']))
  self.assertFalse(model['prefix_state_qualified']);self.assertEqual(model['runtime_python_source_count'],6)
 def test_actual_producer_lists_consumed_positive(self):
  self.assertTrue(self.identity['passed']);self.assertTrue(self.consume()['post_full4_source_qualified'])
  for path,row in zip(self.shards,self.identity['rows']):self.assertEqual(c.stat_signature(path),row['stat_after']);self.assertEqual(len(row['stat_after']),5)
 def test_real_inode_ctime_change_refused_with_mtime_restored(self):
  path=self.shards[2];before=path.stat();mode=before.st_mode&0o777;os.chmod(path,mode^0o100);os.chmod(path,mode)
  os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
  self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns);self.assertNotEqual(path.stat().st_ctime_ns,before.st_ctime_ns)
  with self.assertRaises(ValueError):self.consume()
 def test_legacy_dict4_producer_refused(self):
  row=self.identity['rows'][2];signature=row['stat_after'];legacy=dict(device=signature[0],inode=signature[1],size=signature[2],mtime_ns=signature[3]);row['stat_before']=row['stat_after']=legacy;c.write(self.identitypath,self.identity);self.report['post_model_identity']['sha256']=c.sha(self.identitypath);c.write(self.reportpath,self.report)
  with self.assertRaises(ValueError):self.consume()

if __name__=='__main__':unittest.main()
