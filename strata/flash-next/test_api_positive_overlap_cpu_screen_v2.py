import ast,copy,importlib.util,json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import qualify_api_positive_overlap_cpu_screen_v2 as m

class Tests(unittest.TestCase):
 def response(self,n=8,stop='eos'):
  return dict(stop=True,stop_type=stop,truncated=False,tokens_evaluated=3,prompt='rendered',tokens=[20]*n,tokens_predicted=n,content='An explanation.',generation_settings=dict(temperature=0,seed=1234,n_predict=64,repeat_penalty=1,samplers=['temperature'],ignore_eos=False))
 def test_current_raw_schema_no_invented_flags(self):
  row=self.response();self.assertNotIn('stopped_eos',row);self.assertNotIn('stopped_limit',row)
  self.assertTrue(m.response_gate(row,[1,2,3],0,'rendered'));self.assertTrue(m.continuation_eligible(row))
 def test_short_eos_and_limit_preserved_as_ineligible_measurements(self):
  for row in (self.response(1),self.response(64,'limit')):
   self.assertTrue(m.response_gate(row,[1,2,3],0,'rendered'));self.assertFalse(m.continuation_eligible(row))
 def test_changed_ids_settings_counts_are_not_measurements(self):
  for key,value in [('tokens_predicted',9),('tokens',[True]*8),('tokens_evaluated',4),('stop_type','other'),('truncated',True)]:
   row=self.response();row[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.response_gate(row,[1,2,3],0,'rendered')
  row=self.response();row['generation_settings']['seed']=1
  with self.assertRaises(ValueError):m.response_gate(row,[1,2,3],0,'rendered')
 def candidates(self):return [dict(id='first',target_indices=[0,1]),dict(id='second',target_indices=[2,3]),dict(id='third',target_indices=[4,5])]
 def cases(self):return [dict(case=i,repeat=r,passed=True,response=self.response(),continuation_eligible=True)for i in range(8)for r in (0,1)]
 def test_all16_before_first_declared_selection(self):
  cases=self.cases();self.assertEqual(m.selection_binding(cases,self.candidates())['selected_candidate'],'first')
  with self.assertRaises(ValueError):m.selection_binding(cases[:8],self.candidates())
  cases[4]['response']=self.response(1);cases[4]['continuation_eligible']=False
  cases[5]['response']=self.response(1);cases[5]['continuation_eligible']=False
  self.assertEqual(m.selection_binding(cases,self.candidates())['selected_candidate'],'second')
 def test_no_success_subset_or_ignored_warm_eos(self):
  cases=self.cases();cases[0]['response']=self.response(1);cases[0]['continuation_eligible']=False
  cases[1]['response']=self.response(1);cases[1]['continuation_eligible']=False
  result=m.selection_binding(cases,self.candidates());self.assertIsNone(result['selected_candidate']);self.assertFalse(result['continuation_screen_passed'])
  cases=self.cases();cases[-1]['passed']=False
  with self.assertRaises(ValueError):m.selection_binding(cases,self.candidates())
 def test_duplicate_repeat_mismatch_and_forged_eligibility(self):
  for mutate in ('duplicate','response','eligibility'):
   rows=self.cases()
   if mutate=='duplicate':rows[-1]=rows[-2]
   elif mutate=='response':rows[-1]['response']['tokens'][0]=21
   else:rows[-1]['continuation_eligible']=False
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):m.selection_binding(rows,self.candidates())
 def test_exact8_metadata_and_mount_depth(self):
  messages=[[dict(role='user',content=str(i))]for i in range(8)]
  prepared={'fixtures':[dict(messages=x,rendered='rendered',ids=[1,2,3])for x in messages]}
  self.assertTrue(m.fixture_gate({'corpus_messages':messages},prepared))
  with self.assertRaises(ValueError):m.fixture_gate({'corpus_messages':messages},{'fixtures':prepared['fixtures'][:2]})
  self.assertEqual(Path(m.CONTAINER_RUNNER).parents[2],Path('/harness'))
 def test_lifecycle_memory_identity_helpers_unchanged_AST(self):
  old=ast.parse((m.ROOT/'llamacpp/flash-next/qualify_cpu_functional_pilot_v3.py').read_bytes());new=ast.parse(Path(m.__file__).read_bytes())
  selected=['dependency_binding','inside_server','original_build_gate','model_identity','post_identity','known_page_target','memory_snapshot','memory_gate','OwnedServerStop','monitor_memory','port_preflight']
  defs=lambda tree:{x.name:ast.dump(x,include_attributes=False)for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
  for name in selected:self.assertEqual(defs(old)[name],defs(new)[name],name)
 def test_foreign_owner_and_idempotent_stop(self):
  state={'running':True}
  def inspect(name):return dict(Name='/owned',Image='image',Config={'Labels':{'b70.api-overlap.cpu-screen':'binding'}},State={'Running':state['running'],'ExitCode':0})
  def controlled(*a,**k):state['running']=False;return Mock(returncode=0)
  with patch.object(m,'inspect',side_effect=inspect),patch.object(m.subprocess,'run',side_effect=controlled)as run:
   stop=m.OwnedServerStop('owned','image','binding');self.assertIs(stop.stop(),stop.stop());self.assertEqual(run.call_count,1)
  with patch.object(m,'inspect',return_value=dict(Name='/foreign')),patch.object(m.subprocess,'run')as run:
   with self.assertRaises(ValueError):m.OwnedServerStop('owned','image','binding').stop()
   run.assert_not_called()
 def test_memory_failure_releases_blocked_http(self):
  failed={'host':{'MemAvailable':5,'SwapFree':10}};plan={'minimum_live_available_bytes':6,'memory_cap_bytes':100};base={'host':{'SwapFree':10}}
  released=threading.Event();done=threading.Event();stopping=threading.Event();errors=[];state={'running':True}
  def blocked():released.wait(2);done.set()
  def inspect(name):return dict(Name='/owned',Image='image',Config={'Labels':{'b70.api-overlap.cpu-screen':'binding'}},State={'Running':state['running'],'ExitCode':0})
  def controlled(*a,**k):state['running']=False;released.set();return Mock(returncode=0)
  t=threading.Thread(target=blocked);t.start()
  with patch.object(m,'memory_snapshot',return_value=failed),patch.object(m,'inspect',side_effect=inspect),patch.object(m.subprocess,'run',side_effect=controlled):
   monitor=threading.Thread(target=m.monitor_memory,args=(1,stopping,[],errors,base,plan,m.OwnedServerStop('owned','image','binding')));monitor.start();self.assertTrue(done.wait(1));monitor.join(1)
  released.set();t.join(1);self.assertTrue(errors);self.assertTrue(stopping.is_set())
 def test_post_identity_guards_even_failure(self):
  page=Mock();page.preserve.return_value={'passed':True}
  with patch.object(m,'model_identity',side_effect=ValueError('source changed')):
   with self.assertRaises(ValueError):m.post_identity(page,Path('/third'),'/model',{},Path('/out'),10)
  self.assertEqual(page.preserve.call_count,2)
 def test_actual_command_to_inspection_recipe_and_foreign_mount(self):
  plan=dict(image='image',runner_sha256='binding',memory_cap_bytes=116<<30)
  command=m.server_command('owned',Path('/tmp/case'),plan,Path('/build'),[Path('/model/shard'+str(i))for i in range(4)],Path(m.__file__),None)
  at=command.index('image');mounts=[]
  for i,value in enumerate(command):
   if value=='-v':
    source,dest,mode=command[i+1].rsplit(':',2);mounts.append(dict(Source=source,Destination=dest,RW=mode=='rw',Type='bind'))
  obj=dict(Image='image',Config=dict(Image='image',Cmd=command[at+1:],Entrypoint=['/usr/bin/env'],User=command[command.index('--user')+1],WorkingDir='/results/empty'),HostConfig=dict(NetworkMode='host',Memory=116<<30,MemorySwap=116<<30,NanoCpus=8*10**9,PidsLimit=256,Devices=[],DeviceRequests=None,Privileged=False,GroupAdd=None),Mounts=mounts)
  self.assertTrue(m.runtime_recipe_gate(obj,command))
  for mutate in ('mount','argv','CPU','device'):
   bad=copy.deepcopy(obj)
   if mutate=='mount':bad['Mounts'][-1]['Source']='/foreign-model'
   elif mutate=='argv':bad['Config']['Cmd']=['foreign']
   elif mutate=='CPU':bad['HostConfig']['NanoCpus']=1
   else:bad['HostConfig']['Devices']=[{'PathOnHost':'/dev/dri'}]
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):m.runtime_recipe_gate(bad,command)
 def test_complete_artifact_tree_rejects_response_mutation(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);raw=root/'completion-response.json';m.write(raw,self.response())
   report={'artifact_sha256':{'completion-response.json':m.sha(raw)}}
   self.assertEqual(m.artifact_binding(root,report),report['artifact_sha256'])
   m.write(raw,self.response(1))
   with self.assertRaises(ValueError):m.artifact_binding(root,report)

if __name__=='__main__':unittest.main()
