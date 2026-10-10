"""Synthetic receipt and source admission tests; no Docker/tokenizer/model run."""
import copy,unittest
from pathlib import Path
from unittest.mock import patch
import produce_api_shortwarm_tokenizer_fixture_v1 as q
import generate_api_shortwarm_positive_case_v1 as gen
class Controls(unittest.TestCase):
 def fixtures(self):
  seed=q.read(q.SEED);prior=q.read(q.HERE/'api-cache-positive-case2-source35-v2.json');baseline={'fixtures':{p:[{'logical_index':i,'messages':m,'rendered':'CPU_SYNTHETIC_RENDERED','ids':prior['api_token_ids'][p][i]} for i,m in enumerate(prior['messages'][p])] for p in ('warm','target')}};inputs={'sources':{'tools/strata_tokenizer.py':'CPU_SOURCE_A','serve/frontend.py':'CPU_SOURCE_B'},'tokenizer_files':{'CPU_FIXTURE':'CPU_DIGEST'}};f={'shortwarm_fixture_generation':1,'source_seed_sha256':q.sha(q.SEED),'producer_sha256':q.sha(q.RENDER),'source_file_sha256':{'/'+k:v for k,v in inputs['sources'].items()},'tokenizer_file_sha256':copy.deepcopy(inputs['tokenizer_files']),'actual_GPU_touch':False,'actual_model_payload_read':False,'warm_two_row_runtime_qualified':False,'matched_buffer_only_A_B_input_equivalent':False,'fixtures':copy.deepcopy(baseline['fixtures'])}
  for i,r in enumerate(f['fixtures']['warm']):r.update(messages=seed['messages']['warm'][i],ids=[248045,8678,198,100+i,248045,12,13,14,248045,18,19,20])
  return f,seed,baseline,inputs
 def object(self):
  root=Path('/tmp/CPU_SHORTWARM_MOCK');name,cmd,mounts=q.recipe(root,123);return root,{'Name':'/'+name,'Config':{'Image':q.IMAGE,'Labels':{'b70.api.shortwarm.fixture':q.sha(q.SOURCE_PLAN)},'User':'1000:1000','Entrypoint':['/usr/bin/env'],'Cmd':cmd[cmd.index(q.IMAGE)+1:]},'HostConfig':{'NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,'Memory':512<<20,'MemorySwap':512<<20,'NanoCpus':1000000000,'PidsLimit':128,'Devices':[],'DeviceRequests':None},'Mounts':[{'Source':str(s),'Destination':d,'RW':rw,'Type':'bind'} for s,d,rw in mounts]}
 def test_actual_source_recipe_cpu_only_no_model_devices(self):
  root,obj=self.object();q.owned(obj,root,123);_,cmd,mounts=q.recipe(root,123);self.assertNotIn('--device',cmd);self.assertNotIn('--gpus',cmd);self.assertFalse(any('.gguf' in w for w in cmd));self.assertEqual(cmd[cmd.index('--cpus')+1],'1');self.assertEqual(cmd[cmd.index('--memory')+1],'512m');self.assertFalse(any('/dev/' in str(s) for s,_,_ in mounts))
 def test_foreign_owner_image_label_namespace_refused(self):
  for key in ('name','image','label'):
   root,obj=self.object()
   if key=='name':obj['Name']='/FOREIGN'
   if key=='image':obj['Config']['Image']='FOREIGN'
   if key=='label':obj['Config']['Labels']['b70.api.shortwarm.fixture']='FOREIGN'
   with self.assertRaises(ValueError):q.owned(obj,root,123)
 def test_device_resource_network_mount_and_entrypoint_refused(self):
  for key,value in [('Devices',[{'PathOnHost':'/dev/dri'}]),('NetworkMode','host'),('MemorySwap',-1),('NanoCpus',2000000000),('Privileged',True),('ReadonlyRootfs',False)]:
   root,obj=self.object();obj['HostConfig'][key]=value
   with self.assertRaises(ValueError):q.owned(obj,root,123)
  root,obj=self.object();obj['Mounts'][0]['RW']=True
  with self.assertRaises(ValueError):q.owned(obj,root,123)
  root,obj=self.object();obj['Config']['Entrypoint']=['FOREIGN']
  with self.assertRaises(ValueError):q.owned(obj,root,123)
 def test_target235_exact_and_new_shortwarm_contract(self):
  f,seed,baseline,inputs=self.fixtures();r=q.fixture_contract(f,seed,baseline,inputs);self.assertFalse(r['actual_initial_nonreuse_runtime_qualified'])
 def test_target_messages_ids_rendered_or_type_mutation_refused(self):
  for kind in ('messages','ids','rendered','type'):
   f,seed,b,inputs=self.fixtures()
   if kind=='messages':f['fixtures']['target'][0]['messages'][0]['content']='FOREIGN'
   if kind=='ids':f['fixtures']['target'][0]['ids'][10]+=1
   if kind=='rendered':f['fixtures']['target'][0]['rendered']='FOREIGN'
   if kind=='type':f['fixtures']['warm'][0]['ids'][3]=True
   with self.assertRaises(ValueError):q.fixture_contract(f,seed,b,inputs)
 def test_warm_checkpoint_collision_or_unchanged_length_refused(self):
  for kind in ('same','long'):
   f,seed,b,inputs=self.fixtures();f['fixtures']['warm'][1]['ids']=copy.deepcopy(f['fixtures']['warm'][0]['ids']) if kind=='same' else b['fixtures']['warm'][1]['ids']
   with self.assertRaises(ValueError):q.fixture_contract(f,seed,b,inputs)
 def test_cpu_scope_source_and_tokenizer_negative(self):
  for kind in ('GPU','source','tokenizer'):
   f,seed,b,inputs=self.fixtures()
   if kind=='GPU':f['actual_GPU_touch']=True
   if kind=='source':f['source_file_sha256']['/serve/frontend.py']='FOREIGN'
   if kind=='tokenizer':f['tokenizer_file_sha256']['CPU_EXTRA']='FOREIGN'
   with self.assertRaises(ValueError):q.fixture_contract(f,seed,b,inputs)
 def test_generator_requires_public_fixture_and_preserves_targets(self):
  f,seed,b,inputs=self.fixtures()
  with patch.object(q,'finalized_binding',return_value=(f,{'CPU_ONLY':True})):
   case=gen.case_from_fixture(Path('/tmp/CPU_ONLY'),28692);self.assertEqual(case['api_token_ids']['target'],[r['ids'] for r in b['fixtures']['target']]);self.assertEqual(case['shortwarm_case_generation'],1);self.assertFalse(case['genuine_runtime_preparable']);self.assertFalse(case['matched_buffer_only_A_B_input_equivalent']);self.assertEqual(case['api_warm_max_new_by_request'],[32,32])
   with self.assertRaises(ValueError):gen.case_from_fixture(Path('/tmp/CPU_ONLY'),True)
  with patch.object(q,'finalized_binding',side_effect=ValueError('CPU_BAD_RECEIPT')):
   with self.assertRaises(ValueError):gen.case_from_fixture(Path('/tmp/CPU_ONLY'))
if __name__=='__main__':unittest.main()
