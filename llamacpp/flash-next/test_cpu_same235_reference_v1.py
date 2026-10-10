"""Exact input and natural CPU diagnostics; synthetic responses only, no model."""
import ast,copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('same235_cpu',HERE/'qualify_cpu_same235_reference_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
oldspec=importlib.util.spec_from_file_location('frozen_cpu',HERE/'qualify_cpu_functional_pilot_v3.py');old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)

class Reference(unittest.TestCase):
 def plan(self):return json.loads((HERE/'cpu-same235-reference-source-plan-v1.json').read_bytes())
 def response(self,n=2,finish='eos'):
  p=self.plan();x=p['expected_target_fixtures'][0]
  return {'stop':True,'stop_type':finish,'truncated':False,'tokens_evaluated':235,'prompt':x['rendered'],'tokens':[7]*n,'tokens_predicted':n,'generation_settings':{'temperature':0,'seed':1,'n_predict':64,'repeat_penalty':1,'samplers':['temperature'],'ignore_eos':False},'content':'One'}
 def test_two_exact_multirole235_fixtures(self):
  p=self.plan();self.assertTrue(m.fixture_gate(p,{'fixtures':p['expected_target_fixtures']}));self.assertEqual([len(x['ids']) for x in p['expected_target_fixtures']],[235,235]);self.assertEqual([x[0]['role'] for x in p['prompts']],['system','system']);self.assertNotEqual(p['prompts'][0],p['prompts'][1])
 def test_changed_message_rendered_or_token_refused(self):
  p=self.plan()
  for field in ('messages','rendered','ids'):
   bad=copy.deepcopy(p['expected_target_fixtures']);bad[0][field]=[] if field!='rendered' else 'changed'
   with self.assertRaises(ValueError):m.fixture_gate(p,{'fixtures':bad})
 def test_natural_one_eos_is_observation_not_functional_grade(self):
  p=self.plan();r=self.response();self.assertTrue(m.response_gate(r,p['expected_target_fixtures'][0]['ids'],0,r['prompt']))
 def test_natural_declared64cap_allowed_without_ignoreEOS(self):
  p=self.plan();r=self.response(64,'limit');self.assertTrue(m.response_gate(r,p['expected_target_fixtures'][0]['ids'],0,r['prompt']))
  r['tokens_predicted']=63;r['tokens']=r['tokens'][:63]
  with self.assertRaises(ValueError):m.response_gate(r,p['expected_target_fixtures'][0]['ids'],0,r['prompt'])
 def test_seed_ignoreEOS_input_and_finish_negatives(self):
  p=self.plan();ids=p['expected_target_fixtures'][0]['ids']
  for key,value in [('stop',False),('stop_type','cancel'),('tokens_evaluated',234),('prompt','other'),('tokens',[]),('truncated',True)]:
   r=self.response();r[key]=value
   with self.assertRaises(ValueError):m.response_gate(r,ids,0,p['expected_target_fixtures'][0]['rendered'])
  for key,value in [('seed',1234),('temperature',.1),('ignore_eos',True),('n_predict',65)]:
   r=self.response();r['generation_settings'][key]=value
   with self.assertRaises(ValueError):m.response_gate(r,ids,0,r['prompt'])
 def test_inherited_memory_ownership_libraries_ports_and_teardown_AST(self):
  parse=lambda module:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
  a,b=parse(old),parse(m)
  for name in ['original_build_gate','model_identity','post_identity','known_page_target','memory_snapshot','memory_gate','monitor_memory','owned','inside_server','server_command','wrapper_version_preflight','port_preflight']:
   self.assertEqual(a[name],b[name],name)
  self.assertEqual(self.plan()['server_argv'],json.loads((HERE/'cpu-functional-pilot-source-plan-v3.json').read_bytes())['server_argv'])
 def test_source_settings_fresh_two_and_no_forced_continue(self):
  s=Path(m.__file__).read_text();self.assertIn("'seed':1",s);self.assertIn('for repeat in (0,):',s);self.assertIn("'cache_prompt':False",s);self.assertNotIn('functional(case,response',s);self.assertIn("len(report['cases'])==2",s);self.assertFalse(self.plan()['native_bitwise_authority']);self.assertFalse(self.plan()['underlying_EOS_cause_established'])
 def test_third_shard_exact_and_both_pages_even_failure(self):
  shards=[Path('/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf'%i) for i in range(1,5)];self.assertEqual(m.known_page_target(shards),shards[2]);page=Mock();page.preserve.return_value={'passed':True}
  with patch.object(m,'model_identity',side_effect=ValueError('CPU mock identity failure')):
   with self.assertRaises(ValueError):m.post_identity(page,shards[2],'/model',{},Path('/out'),1)
  self.assertEqual(page.preserve.call_count,2)
 def test_production_wrapper_depth_and_all_entry_modes(self):
  self.assertEqual(Path(m.CONTAINER_RUNNER).parents[2],Path('/harness'));self.assertEqual(self.plan()['container_runner_path'],m.CONTAINER_RUNNER);s=Path(m.__file__).read_text();self.assertIn("CONTAINER_RUNNER,'--inside-tokenizer'",s);self.assertIn("CONTAINER_RUNNER,'--inside-server'",s);self.assertIn("CONTAINER_RUNNER,'--inside-check'",s)
if __name__=='__main__':unittest.main()
