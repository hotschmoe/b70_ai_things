"""Synthetic shape-correct raw topology fixtures; no actual native/model receipts."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cross_topology_prefix35_v1 as q

class PlanTests(unittest.TestCase):
 def setUp(self):
  args=[]
  for flag,value in q.admission.ctrl.base.GEOMETRY.items():args+=[flag,value]
  args+=['--no-prefill-borrow','--spec','2'];common={'args':args,'tokens':{'cases':{'A':list(range(100,108))}},'prefixes':{str(n):list(range(100,100+n)) for n in q.PREFIXES},'max_new':1,'primary_model_name':'hotschmoe-dd','image':'CPU_mock','engine_receipt_sha256':'mock','source_plan_sha256':q.admission.ctrl.PLAN_SOURCE_SHA,'driver_sha256':q.admission.DRIVER_SHA,'token_fixture_sha256':'mock','dependency_sha256':{'CPU':'mock'},'transport':'serial nativeGEN','engine_root':'/tmp/mockSDK'}
  self.left=dict(common,cards=[0]);self.right=copy.deepcopy(common);self.right.update(cards=[0,1],args=args+['--layer-split','32','--split-device','1','--trim-stage-weights']);self.le={'ZE_AFFINITY_MASK':'0','STRATA_SYCL_NATIVE_HC':'1'};self.re=dict(self.le,ZE_AFFINITY_MASK='0,1');self.chain={'generation':{'prompt_verifier_P30_capture35':True}}
 def call(self):return q.matched_plans(self.left,self.right,self.le,self.re,self.chain,self.chain)
 def test_exact_matched_topology_positive(self):self.assertEqual(self.call()['pair_layer_split'],[32,16])
 def test_source_image_math_or_input_mismatch_refused(self):
  for key,value in [('image','foreign'),('engine_receipt_sha256','old'),('prefixes',{}),('tokens',{})]:
   previous=self.right[key];self.right[key]=value
   with self.assertRaises((ValueError,KeyError)):self.call()
   self.right[key]=previous
  self.re['STRATA_QFUSE']='1'
  with self.assertRaises(ValueError):self.call()
 def test_nonmatching_geometry_or_extra_topology_refused(self):
  self.right['args'][self.right['args'].index('--prefill')+1]='128'
  with self.assertRaises(ValueError):self.call()
 def test_duplicate_geometry_refused(self):
  self.left['args']+=['--prefill','64']
  with self.assertRaises(ValueError):self.call()
 def test_eager_even_zero_refused(self):
  self.le['STRATA_VERIFY_EAGER']=self.re['STRATA_VERIFY_EAGER']='0'
  with self.assertRaises(ValueError):self.call()

class MatrixTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.fixtures=[]
  for index,stages in enumerate(({0:(0,48)},{0:(0,32),1:(32,48)})):
   root=cls.root/str(index)/'child/p30_on/p30';root.mkdir(parents=True);frames=[];requests=[]
   captures=root.parent/'captures';captures.mkdir()
   for request,n in enumerate(q.PREFIXES,1):
    ids=list(range(100,100+n));head=captures/('head%d.f32'%n);head.write_bytes(np.full(248320,n,dtype='<f4').tobytes());requests.append({'prefix':n,'raw':{'ids':ids,'output_ids':[111]},'meta':{'logits':[{'request':str(request),'pid':'9','path':str(head),'sha256':q.sha(head),'bytes':993280}]}});windows=[];pos=0
    while pos<n-1:
     count=min(2 if index==0 else 1,n-1-pos);windows.append((pos,count,'prompt_verifier'));pos+=count
    windows.append((n-1,1,'verifier'))
    for pos,count,route in windows:
     for stage,(lo,hi) in stages.items():
      fields=[];d={'schema':2,'route':route,'nonce':q.window_nonce(9,request,pos,count,route),'pid':9,'request':request,'stage':stage,'lb':lo,'le':hi,'first_position':pos,'rows':count,'gen_ids':ids,'row_floats':10240,'source_extent_bytes':count*40960,'row_coverage_complete':True,'normal_dispatch':q.ROUTES[route][1]}
      for layer in range(lo,hi):
       for phase in q.PHASES:
        values=np.stack([np.full(10240,layer+q.PHASES.index(phase)*.125+(pos+r)*.015625,dtype='<f4') for r in range(count)]);path=root/('%d-%d-%d-%s.f32'%(request,pos,layer,phase));path.write_bytes(values.tobytes());fields.append({'layer':layer,'phase':phase,'path':str(path),'sha256':q.sha(path),'bytes':count*40960,'encoding':'LE_F32[rows,4,2560]'})
      frames.append({'binding':d,'fields':fields})
   cls.fixtures.append(({'frames':frames},requests,stages,root))
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_all2160_actualmatrix_rows_split_global_mapping(self):
  left,right=[q.phase_rows(*fixture) for fixture in self.fixtures];self.assertEqual(len(left),2160);self.assertEqual(set(left),set(right))
  self.assertEqual(left[(4,33,'ffn',1)][1]['matrix_rows'],2);self.assertEqual(right[(4,33,'ffn',1)][1]['matrix_rows'],1)
  self.assertTrue(all(np.array_equal(left[key][0],right[key][0]) for key in left))
 def test_missing_duplicate_stale_nonce_wrong_stage_refused(self):
  for mutation in ('missing','duplicate','nonce','stage'):
   capture=copy.deepcopy(self.fixtures[0][0])
   if mutation=='missing':capture['frames'].pop()
   elif mutation=='duplicate':capture['frames'].append(capture['frames'][0])
   elif mutation=='nonce':capture['frames'][0]['binding']['nonce']=0
   else:capture['frames'][0]['binding']['stage']=1
   with self.assertRaises(ValueError):q.phase_rows(capture,*self.fixtures[0][1:])
 def test_changed_current_raw_file_rejected(self):
  capture=copy.deepcopy(self.fixtures[0][0]);capture['frames'][0]['fields'][0]['sha256']='0'*64
  with self.assertRaises(ValueError):q.phase_rows(capture,*self.fixtures[0][1:])
 def test_wrong_rows_or_foreign_input_association_refused(self):
  for key,value in [('rows',2),('gen_ids',[99]),('pid',10),('schema',1)]:
   capture=copy.deepcopy(self.fixtures[0][0]);capture['frames'][0]['binding'][key]=value
   with self.assertRaises(ValueError):q.phase_rows(capture,*self.fixtures[0][1:])
 def test_complete_comparison_reports_4heads_2160rows_and_legitimate_difference(self):
  admissions=[];roots=[]
  for capture,requests,stages,rawroot in self.fixtures:
   runroot=rawroot.parents[2];roots.append(runroot);(rawroot.parent/'requests.json').write_text(json.dumps(requests));prepared=runroot/'prepared';prepared.mkdir(exist_ok=True);(prepared/'server-config.json').write_text(json.dumps({'env':{'CPU':'mock'}}));args=[] if len(stages)==1 else ['--layer-split','32','--split-device','1'];plan={'prepared':str(prepared),'args':args,'prefixes':{str(n):list(range(100,100+n)) for n in q.PREFIXES}};admissions.append(({}, {},plan,{'capture':capture,'chain':{}}))
  head=Path(self.fixtures[1][1][0]['meta']['logits'][0]['path']);prior=head.read_bytes();changed=np.frombuffer(prior,dtype='<f4').copy();changed[17]+=np.float32(.0001);head.write_bytes(changed.tobytes());self.addCleanup(head.write_bytes,prior)
  old_sha=self.fixtures[1][1][0]['meta']['logits'][0]['sha256'];self.fixtures[1][1][0]['meta']['logits'][0]['sha256']=q.sha(head);self.addCleanup(self.fixtures[1][1][0]['meta']['logits'][0].__setitem__,'sha256',old_sha);(head.parent.parent/'requests.json').write_text(json.dumps(self.fixtures[1][1]))
  with patch.object(q,'source_binding',return_value={'CPU':'mock'}),patch.object(q.admission,'finalized_binding',side_effect=admissions),patch.object(q,'matched_plans',return_value={'CPU':'mock'}):result=q.compare_runs(*roots)
  self.assertTrue(result['comparison_completed']);self.assertFalse(result['all_bitwise_equal']);self.assertEqual(result['counts'],{'full_heads':4,'rowvectors':2160});self.assertEqual(result['first_bitwise_difference']['first_bitwise_differing_index'],17);self.assertEqual(result['first_bitwise_difference']['kind'],'head');self.assertFalse(result['numeric_pass_claim']);self.assertIsNone(result['tolerance_gate']);self.assertTrue(all(row['bitwise_equal'] for row in result['rowvectors']))
 def test_numeric_difference_is_reported_without_tolerance(self):
  a=np.linspace(.1,.2,10240,dtype='<f4');b=a.copy();b[11]+=np.float32(1e-5);metrics=q.compare_observation(a,b)
  self.assertFalse(metrics['bitwise_equal']);self.assertEqual(metrics['first_bitwise_differing_index'],11);self.assertGreater(metrics['nmse'],0);self.assertFalse(metrics['numeric_gate_assigned'])
 def test_read_between_hash_and_raw_access_change_refused(self):
  field={'path':str(self.root/'tiny.f32'),'bytes':16};Path(field['path']).write_bytes(np.zeros(4,dtype='<f4').tobytes());field['sha256']='f'*64
  with patch.object(q,'sha',return_value=field['sha256']):
   with self.assertRaises(ValueError):q.raw_vector(field,self.root,1,4)

class AdmissionTests(unittest.TestCase):
 def test_incomplete_genuine_admission_stops_before_array_access(self):
  with patch.object(q,'source_binding',return_value={}),patch.object(q.admission,'finalized_binding',side_effect=ValueError('Incomplete native parent/source4')) as public,patch.object(q,'raw_vector',side_effect=AssertionError('No raw access')):
   with self.assertRaises(ValueError):q.compare_runs(Path('/tmp/left'),Path('/tmp/right'))
   self.assertEqual(public.call_count,1)
 def test_sourceplan_pins_actual_readonly_validator(self):self.assertIn('strata/flash-next/validate_prefix_residual30_final_v4.py',q.source_binding())

if __name__=='__main__':unittest.main()
