import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import produce_api_positive_overlap_corpus_v2 as p
import prepare_api_positive_overlap_cpu_screen_v2 as screen

class Tests(unittest.TestCase):
 def fixture(self):
  seed=p.read(p.SEED);inputs={'sources':{'tools/strata_tokenizer.py':'t','serve/frontend.py':'f'},'tokenizer_files':{'tiny.json':'j'}}
  fixture={'source_seed_sha256':p.sha(p.SEED),'producer_sha256':p.sha(p.RENDER),'overlap_corpus_fixture_generation':2,'source_file_sha256':{'/'+k:v for k,v in inputs['sources'].items()},'tokenizer_file_sha256':inputs['tokenizer_files'],'fixtures':{}}
  for key in ('actual_GPU_touch','actual_model_payload_read','actual_CPU_continuation_observed','actual_GPU_positive_overlap_observed','warm_two_row_runtime_qualified','matched_buffer_only_A_B_input_equivalent'):fixture[key]=False
  for phase in ('warm','target'):fixture['fixtures'][phase]=[{'logical_index':i,'messages':m,'rendered':'CPU test metadata only','ids':[20]*(32 if phase=='warm'else 128)}for i,m in enumerate(seed['messages'][phase])]
  return fixture,seed,inputs
 def test_complete_new_corpus_scope(self):
  fixture,seed,inputs=self.fixture();r=p.fixture_contract(fixture,seed,inputs)
  self.assertFalse(r['actual_CPU_continuation_observed']);self.assertFalse(r['actual_GPU_positive_overlap_observed'])
  self.assertEqual(len(seed['candidates']),3)
  for row in seed['messages']['target']:
   self.assertNotIn('until the client',row[-1]['content']);self.assertIn('one sentence of 12-20 words',row[-1]['content'])
 def test_missing_changed_and_short_target_rejected(self):
  fixture,seed,inputs=self.fixture()
  for phase in ('warm','target'):
   bad=copy.deepcopy(fixture);bad['fixtures'][phase].pop()
   with self.assertRaises(ValueError):p.fixture_contract(bad,seed,inputs)
  for key,value in [('ids',[20]*64),('ids',[False]*128),('rendered','\u03bb'),('messages',[])]:
   bad=copy.deepcopy(fixture);bad['fixtures']['target'][0][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):p.fixture_contract(bad,seed,inputs)
 def test_tokenization_never_implies_continuation(self):
  fixture,seed,inputs=self.fixture();fixture['actual_CPU_continuation_observed']=True
  with self.assertRaises(ValueError):p.fixture_contract(fixture,seed,inputs)
 def test_exact_candidate_order(self):
  fixture,seed,inputs=self.fixture();seed['candidates'][0]['target_indices']=[2,3]
  with self.assertRaises(ValueError):p.fixture_contract(fixture,seed,inputs)
 def test_recipe_cpu_only_and_no_model_mount(self):
  with patch.object(p,'sha',return_value='a'*64):
   name,cmd,mounts=p.recipe(Path('/tmp/new-corpus'),42)
  self.assertIn('positive-overlap',name);self.assertIn(p.IMAGE,cmd);self.assertNotIn('--device',cmd)
  self.assertNotIn('--gpus',cmd);self.assertEqual(cmd[cmd.index('--network')+1],'none')
  self.assertEqual(cmd[cmd.index('--cpus')+1],'1');self.assertEqual(cmd[cmd.index('--memory')+1],'512m')
  self.assertEqual(cmd[cmd.index('--memory-swap')+1],'512m')
  self.assertEqual([d for s,d,r in mounts if r],['/results'])
  self.assertFalse(any('gguf'in str(s).lower()for s,d,r in mounts))
 def test_declared_screen_exact_requests_and_no_selection(self):
  fixture,seed,inputs=self.fixture()
  with patch.object(p,'source_binding',return_value={'SOURCE_CPU_TEST_ONLY':True}):plan=screen.screen_plan(fixture,{'CPU_TEST_ONLY':True},seed)
  self.assertEqual(len(plan['requests']),8);self.assertIsNone(plan['selected_candidate'])
  self.assertFalse(plan['actual_CPU_screen_observed'])
  for row in plan['requests']:
   self.assertEqual(row['request']['prompt'],row['accepted_input_ids']);self.assertFalse(row['request']['cache_prompt']);self.assertEqual(row['fresh_process_repeats'],2)
 def test_natural_continuation_is_scheduling_predicate(self):
  row=dict(tokens=[20]*8,stop=True,stop_type='eos',truncated=False,content='A complete answer.')
  self.assertTrue(screen.continuation_predicate(row))
  for key,value in [('tokens',[20]),('tokens',[20]*64),('tokens',[True]*8),('stop_type','limit'),('content','')]:
   with self.subTest(key=key):self.assertFalse(screen.continuation_predicate(dict(row,**{key:value})))

if __name__=='__main__':unittest.main()
