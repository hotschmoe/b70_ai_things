"""Source/config-only routing and synthetic owned-state controls; no model payload."""
import ast,copy,gc,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import full48_owned_composition_storage_v2 as route
import full48_owned_composition_storage_v1 as old
import explore_full48_original_routes_v1 as explorer
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
import test_ple_prompt35_cpu_v1 as f35
HERE=Path(__file__).resolve().parent

def config(pair=False):
 launch=json.loads((HERE/'native-hc-launch-plan.json').read_text());args=list(launch['engine_arguments_common']);env=dict(launch['runtime_environment_common']);env.update(STRATA_STAGE_MIRRORS='1',STRATA_STAGE_MIRROR_SEGMENT_MIB='1024')
 for key,value in {'--max-context':'2048','--prefill':'64','--ple-row-cache':'65536','--prompt-cache':'3'}.items():args[args.index(key)+1]=value
 args+=['--adapt-every','0','--no-prefill-borrow','--prompt-cache-root','1','--prompt-cache-every','1000000000','--turn-token','248045']
 if pair:args+=['--layer-split','32','--split-device','1','--trim-stage-weights']
 return args,env

class ScheduleTests(unittest.TestCase):
 def test_normal_source35_T1_T2_schedule_allprefixes(self):
  args,env=config()
  for n,want in [(1,[(0,1)]),(2,[(0,1),(1,1)]),(4,[(0,2),(2,1),(3,1)]),(8,[(0,2),(2,2),(4,2),(6,1),(7,1)])]:
   result=route.derive_schedule([19+i for i in range(n)],args,env);self.assertEqual([(w['position'],w['rows']) for w in result['windows']],want);self.assertTrue(all(w['math_route']=='verifier' for w in result['windows']));self.assertFalse(result['native_route_metadata_used']);self.assertFalse(result['native_window_group_rounding_emulated'])
 def test_pair_topology_same_math_schedule_and_explicit_split_groups(self):
  args,env=config(True);normal=route.derive_schedule([19,20,21,22],args,env);split=route.derive_schedule([19,20,21,22],args+['--spec-split'],env)
  self.assertEqual([w['math_route'] for w in normal['windows']],[w['math_route'] for w in split['windows']]);self.assertEqual([w['groups'] for w in normal['windows']],[1,1,1]);self.assertEqual([w['groups'] for w in split['windows']],[2,1,1]);self.assertFalse(split['native_window_group_rounding_emulated'])
 def test_changed_shape_cache_geometry_sourceflags_fail_closed(self):
  args,env=config()
  for key,value in [('--spec','3'),('--prompt-cache','0'),('--kv','q8'),('--max-context','8192'),('--prefill','128')]:
   changed=list(args);changed[changed.index(key)+1]=value
   with self.assertRaises(ValueError):route.derive_schedule([19,20],changed,env)
  for key,value in [('STRATA_BATCH_PUBLIC_PREFIX','1'),('STRATA_HC_Q8','1'),('STRATA_DEC_BATCH','0'),('STRATA_QFUSE','1')]:
   with self.assertRaises(ValueError):route.derive_schedule([19,20],args,{**env,key:value})
 def test_presence_reread_pipeline_eager_zero_also_rejected(self):
  args,env=config()
  for key in ('STRATA_CKPT_REREAD','STRATA_PIPELINE_SWITCH','STRATA_VERIFY_EAGER','STRATA_GDN_REC_HEADS'):
   for value in ('0','1',''):
    with self.assertRaises(ValueError):route.derive_schedule([19,20],args,{**env,key:value})
 def test_unknown_argv_yarn_and_unknownSTRATA_fail_closed(self):
  args,env=config()
  for suffix in (['--yarn-scale','2'],['--unknown-option','0'],['--pipeline-windows','0'],['--conversation-cache-mib','1'],['--short-read','0']):
   with self.assertRaises(ValueError):route.derive_schedule([19,20],args+suffix,env)
  with self.assertRaises(ValueError):route.derive_schedule([19,20],args,{**env,'STRATA_UNKNOWN_MATH':'0'})
 def test_duplicate_or_ambiguous_flags_rejected(self):
  args,env=config()
  for suffix in (['--spec','2'],['--no-prefill-borrow'],['--spec-split','--no-spec-split']):
   with self.assertRaises(ValueError):route.derive_schedule([19,20],args+suffix,env)
 def test_internal_turncut_unsupported_but_firstturn_allowed(self):
  args,env=config();self.assertEqual(len(route.derive_schedule([248045,19],args,env)['windows']),2)
  with self.assertRaises(ValueError):route.derive_schedule([19,248045],args,env)
 def test_no_observed_route_argument_or_math_input_API(self):
  args,env=config()
  with self.assertRaises(TypeError):route.derive_schedule([19],args,env,native_routes=['prefill'])
  tree=ast.parse(Path(route.__file__).read_text());tokens=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='tokens');self.assertEqual([a.arg for a in tokens.args.args],['self','token_ids']);source=ast.unparse(tokens);self.assertNotIn('captured',source.split('return',1)[0]);self.assertIn('derive_schedule(ids, self.args, dict(self.env))',source)

class OwnedTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):f35.SourceTests.setUpClass();cls.source=f35.SourceTests.source
 @classmethod
 def tearDownClass(cls):f35.SourceTests.tearDownClass()
 def tearDown(self):gc.collect()
 def model(self,provider=None):
  args,env=config();return route.RouteAwareFull48OwnedComposition(provider or SyntheticOriginalRoles(),'a'*64,args,env,self.source)
 def test_exact_source3_binding(self):
  binding=route.bind_dispatch_source(self.source);self.assertEqual(set(binding['files']),set(route.CODE));self.assertFalse(binding['actual_GPU_dispatch_witnessed'])
 def test_wrong_source_before_original_rows(self):
  provider=SyntheticOriginalRoles()
  with tempfile.TemporaryDirectory() as name:
   root=Path(name)
   for rel in route.CODE:path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.source/rel,path)
   (root/route.CODE[0]).write_text('wrong source only')
   args,env=config()
   with self.assertRaises(ValueError):route.RouteAwareFull48OwnedComposition(provider,'a'*64,args,env,root)
   self.assertEqual(provider.row_reads,0)
 def test_prefix1_replays_old_reference_bitwise_allphases_head(self):
  a=old.Full48OwnedComposition(SyntheticOriginalRoles(),'a'*64).tokens([19]);b=self.model().tokens([19])
  for x,y in zip(a['trace'][0]['layers'],b['trace'][0]['layers']):
   for phase in ('input','attention','ffn'):self.assertTrue(np.array_equal(x[phase],y[phase]),(x['layer'],phase))
  self.assertTrue(np.array_equal(a['first_generated_logits'],b['first_generated_logits']));self.assertFalse(b['full_model_math_qualified']);self.assertFalse(b['actual_window_group_shape_rounding_qualified'])
 def test_prefix2_earlier_verifier_storage_not_old_prefill(self):
  result=self.model().tokens([19,22]);self.assertEqual([row['route'] for row in result['trace']],['verifier','verifier']);self.assertEqual([row['normal_dispatch'] for row in result['trace']],['prompt_verify','target_verify']);self.assertEqual(result['last_two_owned'],[19,22]);self.assertFalse(result['captured_inputs_used']);self.assertFalse(result['native_routes_used_as_math_inputs'])
 def test_owned_state_interleaving_and_reset_isolation(self):
  model=self.model();a=model.tokens([19,22]);head=a['first_generated_logits'].copy();state=a['gdn_states_owned'][0]['recurrent'].copy();b=model.tokens([22,19]);self.assertFalse(np.array_equal(head,b['first_generated_logits']));self.assertTrue(np.array_equal(state,a['gdn_states_owned'][0]['recurrent']));del b;gc.collect();again=model.tokens([19,22]);self.assertTrue(np.array_equal(head,again['first_generated_logits']));self.assertTrue(np.array_equal(state,again['gdn_states_owned'][0]['recurrent']))
 def test_cfg_copies_cannot_be_mutated_by_caller(self):
  args,env=config();model=route.RouteAwareFull48OwnedComposition(SyntheticOriginalRoles(),'a'*64,args,env,self.source);args[args.index('--spec')+1]='8';env['STRATA_QFUSE']='1';result=model.tokens([19]);self.assertEqual(result['declared_schedule']['windows'][0]['rows'],1)

class NativeComparisonFixtures(unittest.TestCase):
 @classmethod
 def setUpClass(cls):f35.CollectorTests.setUpClass()
 @classmethod
 def tearDownClass(cls):f35.CollectorTests.tearDownClass()
 def setUp(self):
  fixture=f35.CollectorTests('test_new_truthful_route_allrows_two_stages');fixture.setUp();self.fixture=fixture;self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);directory=self.root/'child/p30_on';shutil.copytree(fixture.root,directory/'p30');(directory/'captures').mkdir();head=directory/'captures/head.f32';head.write_bytes(np.zeros(248320,dtype='<f4').tobytes());self.head={'pid':'9','request':'1','path':str(head),'sha256':explorer.sha(head)}
  capture=fixture.collect()
  for frame in capture['frames']:
   for field in frame['fields']:field['path']=str(directory/'p30'/Path(field['path']).name)
  self.capture=capture;self.plan={'prefixes':{'4':list(fixture.ids)}};rows=[{'prefix':n,'raw':{'ids':fixture.ids if n==4 else [19]*n,'fresh':1},'meta':{'logits':[self.head],'coverage':{'required_stage_ranges':{'0':[0,32],'1':[32,48]}}}} for n in (1,2,4,8)];(directory/'requests.json').write_text(json.dumps(rows));args,env=config();self.schedule=route.derive_schedule(fixture.ids,args,env)
  self.mock=patch.object(explorer,'finalized_binding',return_value=({}, {}, self.plan, {'capture':self.capture}));self.mock.start();self.addCleanup(self.mock.stop)
 def test_allactualrows144N_plus_head_without_native_math_inputs(self):
  ids,values,binding=explorer.native_prefix(self.root,self.plan,4,self.schedule);self.assertEqual(ids,self.fixture.ids);self.assertEqual(len(values),577);self.assertEqual(values['p1_l0_input'].shape,(4,2560));self.assertFalse(binding['native_inputs_or_routes_used_for_own_computation']);self.assertEqual(values['head'].shape,(248320,))
 def test_observed_route_mismatch_not_used_as_schedule(self):
  self.capture['frames'][0]['binding']['route']='prefill'
  with self.assertRaises(ValueError):explorer.native_prefix(self.root,self.plan,4,self.schedule)
 def test_actual_multimatrix_mutation_rejected(self):
  path=Path(self.capture['frames'][0]['fields'][0]['path']);path.write_bytes(bytes(path.stat().st_size))
  with self.assertRaisesRegex(ValueError,'path/SHA/extent'):explorer.native_prefix(self.root,self.plan,4,self.schedule)
 def test_raw_matrix_changed_between_hash_and_read_rejected(self):
  target=Path(self.capture['frames'][0]['fields'][0]['path']);original=Path.read_bytes;count=[0]
  def reads(path):
   raw=original(path)
   if path==target:
    count[0]+=1
    if count[0]==2:raw=bytes([raw[0]^1])+raw[1:]
   return raw
  with patch.object(Path,'read_bytes',reads),self.assertRaisesRegex(ValueError,'matrix changed during read'):explorer.native_prefix(self.root,self.plan,4,self.schedule)
 def test_fullhead_changed_between_hash_and_read_rejected(self):
  target=Path(self.head['path']);original=Path.read_bytes;count=[0]
  def reads(path):
   raw=original(path)
   if path==target:
    count[0]+=1
    if count[0]==2:raw=bytes([raw[0]^1])+raw[1:]
   return raw
  with patch.object(Path,'read_bytes',reads),self.assertRaisesRegex(ValueError,'fullhead changed during read'):explorer.native_prefix(self.root,self.plan,4,self.schedule)
 def test_comparison_is_exploratory_and_preserves_native_head(self):
  _,native,_=explorer.native_prefix(self.root,self.plan,4,self.schedule);owned={'ids':self.fixture.ids,'captured_inputs_used':False,'captured_states_or_selected_ids_used':False,'native_routes_used_as_math_inputs':False,'state_initialized_from_zero':True,'full_model_math_qualified':False,'trace':[{'position':pos,'layers':[{'layer':layer,**{phase:native['p%d_l%d_%s'%(pos,layer,phase)] for phase in ('input','attention','ffn')}} for layer in range(48)]} for pos in range(4)],'first_generated_logits':np.zeros(248320),'gdn_states_owned':{},'qsa_states_owned':{},'ple_history_owned':np.zeros((9,10240)),'last_two_owned':[103,104],'declared_schedule':self.schedule};output=self.root/'comparison';output.mkdir();result=explorer.save_owned_and_compare(output,owned,native);self.assertEqual(len(result['comparisons']),577);self.assertFalse(result['numeric_pass_claim']);self.assertIsNone(result['tolerance_gate']);self.assertEqual(result['arrays']['native_head']['bytes'],993280)

if __name__=='__main__':unittest.main()
