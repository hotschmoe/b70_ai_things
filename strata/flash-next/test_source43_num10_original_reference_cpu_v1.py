import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import source43_num10_original_reference_v1 as q
class Model:
 def __init__(self):self.p=SimpleNamespace(rows=lambda n,ids:np.array([[1.,2.,3.]]));self.g=SimpleNamespace(streams=2,embd=3);self.before=[]
 def initial_state(self):return {'recurrent':np.zeros((2,2,2)),'conv':np.zeros((3,8))}
 def hc_read(self,r,half):return {'normalized':r.copy(),'low_silu':np.zeros(2),'gate':np.zeros_like(r),'inject':np.zeros(2),'mixed':r[0].copy()}
 def mixer(self,x,state):
  self.before.append({k:v.copy()for k,v in state.items()});state['recurrent'][:]=1
  return state,np.array(x),{'qkv':np.zeros(8),'log_decay':np.zeros(2),'beta':np.zeros(2),'q':np.zeros(2),'k':np.zeros(2),'v':np.zeros(4),'gated':np.zeros(4)}
 def hc_write(self,r,b,i):return r+b
 def project(self,n,x):return np.zeros(4)
class Controls(unittest.TestCase):
 def test_independent_has_no_target_argument(self):
  model=Model();a=q.independent_first(model);b=q.independent_first(model);self.assertEqual(a['residual_input'].tobytes(),b['residual_input'].tobytes());self.assertTrue(all(not np.any(s['recurrent'])and not np.any(s['conv'])for s in model.before));self.assertTrue(np.all(a['gdn_state_after']==1))
 def test_shapes_exact_existing_NUM10_software(self):
  contract=q.field_contract();self.assertEqual(len(contract),33)
  for f in contract:
   if f['name']in q.SHAPES:self.assertEqual(f['bytes'],int(np.prod(q.SHAPES[f['name']]))*4)
 def test_target_content_mutation(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'data';p.write_bytes(b'abcd');want=hashlib.sha256(b'abcd').hexdigest();self.assertEqual(q.consume(p,4,want),b'abcd');p.write_bytes(b'bcde')
   with self.assertRaises(ValueError):q.consume(p,4,want)
 def test_alias_before_read(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'data').write_bytes(b'abcd');(p/'alias').symlink_to(p/'data')
   with self.assertRaises(ValueError):q.consume(p/'alias',4,hashlib.sha256(b'abcd').hexdigest())
 def test_no_old_candidate_runtime_imports(self):
  text=Path(q.__file__).read_text();self.assertNotIn('candidate_binding',text);self.assertNotIn('validate_layer0_numerical_final_v10',text);self.assertNotIn('C113',text)
 def test_native_state_not_conditional_input(self):
  model=Model();targets={'attn_mixed':np.ones(3),'gdn_output_gated':np.zeros(6144,dtype='<f4')};r=q.conditional(model,targets);self.assertFalse(r['native_HC_mixed_to_original_zero_GDN']['captured_state_used']);self.assertTrue(all(not np.any(s['recurrent'])for s in model.before))
 def test_actor_scope_refused_before_raw_read(self):
  b={'source_generation':41,'materialized_mode':1,'NUM10_enabled':True,'QSA3_enabled':False}
  with self.assertRaises(ValueError):q.load_targets(b,[])
 def test_metrics_no_tolerance(self):self.assertFalse(q.metrics(np.ones(3),np.ones(3))['numeric_gate_assigned'])
if __name__=='__main__':unittest.main()

class RawControls(unittest.TestCase):
 def fixture(self,root,kind):
  d=root/'layer0';d.mkdir();rows=[];contract=q.field_contract();pid=17
  for request in range(1,5):
   fields=[];observed=[]
   for w in contract:
    p=d/('%d-%s.bin'%(request,w['name']));data=bytes(w['bytes'])
    if kind=='nonzero'and request==1 and w['name']=='gdn_conv_before':data=b'\x01'+data[1:]
    if kind=='negative_zero'and request==1 and w['name']=='gdn_state_before':data=b'\0\0\0\x80'+data[4:]
    p.write_bytes(data);fields.append(dict(w,file=p.name));observed.append(dict(w,path=str(p),sha256=hashlib.sha256(data).hexdigest()))
   f={'schema':1,'request':request,'pid':pid,'binding_sha256':'b'*64,'position':0,'gen_ids':[248045],'token':248045,'rows':1,'layer':0,'stage':0,'reused':0,'completed_nonce':(pid<<32)|request,'request_replay_marker_verified':True,'producer_mapping_verified':True,'complete_preregistered_layout':True,'full_model_math_qualified':False,'raw_fused_hidden_observed':False,'graph_key':2,'fields':fields};meta=d/('%d.json'%request);meta.write_text(json.dumps(f));row={'raw':{'ids':[248045],'fresh':1,'pin':None},'layer0':{'frame':f,'metadata':str(meta),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),'observed':observed}};rows.append(row)
  if kind=='wrongtoken':rows[0]['raw']['ids']=[8678]
  if kind=='fresh':rows[0]['raw']['fresh']=0
  return {'source_generation':43,'materialized_mode':1,'NUM10_enabled':True,'QSA3_enabled':False,'capture_root':str(root),'engine_pid':pid,'plan_sha256':'b'*64},rows
 def run_case(self,kind):
  with tempfile.TemporaryDirectory()as t:
   b,rows=self.fixture(Path(t),kind)
   if kind=='valid':self.assertEqual(len(q.load_targets(b,rows)),4)
   else:
    with self.assertRaises(ValueError):q.load_targets(b,rows)
 def test_actual_shaped_raw33(self):self.run_case('valid')
 def test_nonzero_native_conv(self):self.run_case('nonzero')
 def test_native_negative_zero_not_initial_bytezero(self):self.run_case('negative_zero')
 def test_wrong_actual_accepted_token(self):self.run_case('wrongtoken')
 def test_nonfresh_actual_request(self):self.run_case('fresh')

class ClosingControls(unittest.TestCase):
 def test_original_failure_still_attempts_both_closings(self):
  calls=[]
  def capture(root):calls.append('capture');return {'requests':[]}
  def scan(t):calls.append('scan');raise ValueError('actual failed scan')
  with patch.object(q,'source_binding',return_value='source'),patch.object(q,'load_targets',return_value=[]):
   with self.assertRaisesRegex(ValueError,'original failed'):
    q.enclosed('root',lambda:(_ for _ in()).throw(ValueError('original failed')),capture,scan)
  self.assertEqual(calls,['capture','capture','scan'])
 def test_raw_failure_still_attempts_both_closings(self):
  calls=[]
  def capture(root):calls.append('capture');return {'requests':[]}
  def scan(t):calls.append('scan');raise ValueError('actual failed scan')
  with patch.object(q,'source_binding',return_value='source'),patch.object(q,'load_targets',side_effect=ValueError('raw failed')):
   with self.assertRaisesRegex(ValueError,'raw failed'):q.enclosed('root',lambda:None,capture,scan)
  self.assertEqual(calls,['capture','capture','scan'])
 def test_closing_capture_failure_does_not_skip_publisher_attempt(self):
  calls=[]
  def capture(root):
   calls.append('capture')
   if len(calls)>1:raise ValueError('actual capture closing failed')
   return {'requests':[]}
  def scan(t):calls.append('scan');raise ValueError('actual failed scan')
  with patch.object(q,'source_binding',return_value='source'),patch.object(q,'load_targets',side_effect=ValueError('raw failed')):
   with self.assertRaisesRegex(ValueError,'actual capture closing failed'):q.enclosed('root',lambda:None,capture,scan)
  self.assertEqual(calls,['capture','capture','scan'])
