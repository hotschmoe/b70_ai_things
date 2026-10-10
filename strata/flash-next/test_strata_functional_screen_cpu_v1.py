#!/usr/bin/env python3
"""Meaningful semantics/natural64/fixtures/repeats adapter controls only."""
import ast,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import strata_functional_screen_v1 as q
import qualify_strata_functional_screen_v1 as parent

class FunctionalTests(unittest.TestCase):
 def raw(self):
  ids=[248045,19,20];return {'ids':ids,'fresh':1,'cancel_requested':None,'stop_sent':False,'done':'DONE 18 3 0 0 stop','output_ids':[19]*18,'command':q.serial.request_command(ids,64,1,None)}
 def test_exact_same_CPU_semantic_checker_code_and_arithmetic(self):
  raw=self.raw();self.assertTrue(q.functional_request(raw,0,'def add(a, b):\n    return a + b\n',raw['ids'])['passed']);raw['output_ids']=[19]*3;raw['done']='DONE 3 3 0 0 stop';self.assertTrue(q.functional_request(raw,1,'12',raw['ids'])['passed'])
 def test_wrong_semantic_output_extra_reasoning_prose_and_code_refused(self):
  for text in ('<think>x</think>\ndef add(a,b): return a+b','Here is code:\ndef add(a,b): return a+b','def add(a,b): return a-b','import os\ndef add(a,b): return a+b'):
   with self.assertRaises((ValueError,SyntaxError)):q.functional_request(self.raw(),0,text,self.raw()['ids'])
  raw=self.raw();raw['output_ids']=[19]*3;raw['done']='DONE 3 3 0 0 stop'
  for text in ('13','The answer is12','12 balls'):
   with self.assertRaises(ValueError):q.functional_request(raw,1,text,raw['ids'])
 def test_length64_cancel_or_partialcount_refused(self):
  for update in ({'done':'DONE 18 3 0 0 length'},{'done':'DONE 18 3 0 0 cancel'},{'output_ids':[19]*64,'done':'DONE 64 3 0 0 stop'},{'done':'DONE 17 3 0 0 stop'},{'done':'DONE 18 2 0 0 stop'}):
   raw=self.raw();raw.update(update)
   with self.assertRaises(ValueError):q.functional_request(raw,0,'def add(a,b): return a+b',self.raw()['ids'])
 def test_fresh_command_greedy_budget_and_identity_refused(self):
  for update in ({'fresh':0},{'command':q.serial.request_command([248045,19,20],32,1,None)},{'ids':[248045,20,19]},{'stop_sent':True},{'output_ids':[248320]*18}):
   raw=self.raw();raw.update(update)
   with self.assertRaises(ValueError):q.functional_request(raw,0,'def add(a,b): return a+b',self.raw()['ids'])
 def test_actual_CPU_fixture_metadata_only_and_exact_successful_artifacts(self):
  path=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v3-20261010/tokenizer-fixtures.json');fixture=q.read(path);q.fixture_gate(q.read(q.CPU_PLAN),fixture);binding=q.cpu_run_binding(path);self.assertEqual(binding['input_prompt_lengths'],[30,39]);self.assertTrue(binding['functional_fresh_repeats_observed']);self.assertFalse(binding['full_model_math_qualified']);self.assertFalse(binding['native_bitwise_authority'])
  bad=copy.deepcopy(fixture);bad['fixtures']=bad['fixtures'][::-1]
  with self.assertRaises(ValueError):q.fixture_gate(q.read(q.CPU_PLAN),bad)
 def test_old_or_observer_on_functional_plan_refused_before_baseline(self):
  with patch.object(q.proof,'genuine_baseline') as baseline:
   with self.assertRaises(ValueError):q.manifest_binding({'screen_generation':0})
   baseline.assert_not_called()
 def test_parent_reuses_same_lifecycle_no_broad_fork(self):
  source=Path(parent.__file__).read_text();self.assertIn('import qualify_batch_numerical_v7 as lifecycle',source);self.assertIn('return lifecycle.main()',source);self.assertNotIn('def health(',source);self.assertNotIn('Docker',source)
 def test_actual_parent_adapter_delegation_restores_processlocal_state_in_test(self):
  template=parent.lifecycle;original_file=template.__file__;original_ctrl=template.ctrl;original_sha=template.BATCH_NUMERICAL_SHA
  with patch.object(template,'__file__',original_file),patch.object(template,'ctrl',original_ctrl),patch.object(template,'BATCH_NUMERICAL_SHA',original_sha),patch.object(template,'main',return_value=0) as shared:
   self.assertEqual(parent.main(),0);shared.assert_called_once();self.assertIs(template.ctrl,q);self.assertEqual(template.__file__,parent.__file__);self.assertEqual(template.BATCH_NUMERICAL_SHA,q.sha(Path(q.__file__)))
  self.assertEqual(template.__file__,original_file);self.assertIs(template.ctrl,original_ctrl)
 def test_changed_CPU_report_and_optimized_Python_refused(self):
  fixtures=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v3-20261010/tokenizer-fixtures.json');realsha=q.sha
  def wrong(path):return 'a'*64 if Path(path).name=='report.json' else realsha(path)
  with patch.object(q,'sha',wrong):
   with self.assertRaises(ValueError):q.cpu_run_binding(fixtures)
  with patch.dict(q.os.environ,{'PYTHONOPTIMIZE':'1'}):
   with self.assertRaisesRegex(ValueError,'assertions enabled'):q.manifest_binding({})
 def test_exact_preparedcfg_recipe_rejects_unknownmath_flags_args_image_pack(self):
  from test_full48_route_reference_cpu_v2 import config
  args,env=config();cfg={'args':args,'env':env};derived_args,derived_env=q.config_contract(cfg);metadata={'runtime':{'image':'PINNED_IMAGE'},'pack':'PINNED_PACK'};plan={'args':derived_args,'env':derived_env,'image':'PINNED_IMAGE','pack':'PINNED_PACK'};self.assertTrue(q.exact_config_gate(plan,metadata,cfg))
  for key,value in [('args',derived_args+['--unknown-math','1']),('env',dict(derived_env,STRATA_UNKNOWN_MATH='1')),('image','FOREIGN_IMAGE'),('pack','FOREIGN_PACK')]:
   changed=copy.deepcopy(plan);changed[key]=value
   with self.assertRaises(ValueError):q.exact_config_gate(changed,metadata,cfg)
 def test_native_command_recipe_is_exact_and_recollected(self):
  from types import SimpleNamespace
  import os
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);directory=root/'native';directory.mkdir();q.write(root/'plan.snapshot.json',{'CPU_SYNTHETIC':True});plan={'env':{'STRATA_SYCL_NATIVE_HC':'1'},'engine_root':'/CPU_FAKE_SDK','pack':'/CPU_FAKE_PACK','image':'CPU_IMAGE','args':['--max-context','2048']};realstat=os.stat
   def stat(path,*a,**kw):return SimpleNamespace(st_gid=123) if str(path)=='/dev/dri/renderD128' else realstat(path,*a,**kw)
   with patch.object(q.os,'stat',stat):command,container=q.native_command(plan,directory,424242)
   self.assertEqual(container,'b70-prefix-424242-functional');self.assertIn('123',command);self.assertIn('STRATA_PREFIX_DIAG=1',command);self.assertIn('STRATA_PREFIX_LIFECYCLE_DIAG=0',command);self.assertIn('b70.prefix.plan='+q.sha(root/'plan.snapshot.json'),command);self.assertIn('CPU_IMAGE',command)
  reader=(Path(q.__file__).with_name('validate_strata_functional_screen_v1.py')).read_text();self.assertIn('ctrl.native_command(',reader);self.assertIn('post_raw_source_join',reader);self.assertGreaterEqual(reader.count('final_source_join(root,parent,plan,child)'),2)
 def test_no_registry_write_or_backend_math_change(self):
  source=Path(q.__file__).read_text();self.assertIn("'registry_edit_applied':False",source);self.assertIn("'registered_quality_qualified':False",source);self.assertIn("'fresh_scope':'allstage explicit freshGEN reset",source);self.assertNotIn('models.yaml',source);self.assertNotIn('imagegen',source)
if __name__=='__main__':unittest.main()
