import copy,hashlib,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
import audit_source43_zero_gdn_state_v1 as q
import test_source43_q81_boundaries_cpu_v1 as previous
class Controls(unittest.TestCase):
 def fixture(self,S=4,HK=2,HV=4):
  h=np.concatenate([np.zeros(HK*S,dtype='<f4'),np.arange(1,HK*S+1,dtype='<f4')*.125,np.arange(1,HV*S+1,dtype='<f4')*.25]);a=np.stack([np.full(HV,-1,dtype='<f4'),np.full(HV,.5,dtype='<f4')]);state=np.empty((S,HV,S),dtype='<f4')
  for i in range(S):
   for head in range(HV):
    for j in range(S):state[i,head,j]=np.float32(np.float32(h[HK*S+(head%HK)*S+i])*np.float32(h[2*HK*S+head*S+j]*a[1,head]))
  return h.tobytes(),a.tobytes(),bytes(S*HV*S*4),state.tobytes()
 def check(self,values):return q.state_comparison(*values,S=4,HK=2,HV=4)
 def test_exact_independent_scalar_mapping(self):self.assertTrue(self.check(self.fixture())['state_exact'])
 def test_q_does_not_change_zero_state(self):
  h,a,b,c=self.fixture();h=bytearray(h);h[:32]=struct.pack('<8f',*range(8));self.assertTrue(self.check((bytes(h),a,b,c))['state_exact'])
 def test_wrong_head_mapping_and_column_layout_rejected(self):
  h,a,b,c=self.fixture();state=np.frombuffer(c,dtype='<f4').reshape(4,4,4);bad=state[:,[1,0,2,3],:].copy();self.assertFalse(self.check((h,a,b,bad.tobytes()))['state_exact']);bad=state.transpose(1,0,2).copy();self.assertFalse(self.check((h,a,b,bad.tobytes()))['state_exact'])
 def test_signed_zero_fma_and_before_negative_zero(self):
  h,a,b,c=self.fixture();v=np.frombuffer(h,dtype='<f4').copy();v[16:]=-0.;self.assertTrue(self.check((v.tobytes(),a,b,bytes(len(c))))['state_exact']);bad=bytearray(b);bad[3]=128
  with self.assertRaises(ValueError):self.check((h,a,bytes(bad),c))
 def test_nonfinite_decay_beta_and_bounds(self):
  h,a,b,c=self.fixture()
  for index,value in((0,.1),(0,float('nan')),(4,1.1),(4,-.1)):
   x=np.frombuffer(a,dtype='<f4').copy();x[index]=value
   with self.assertRaises(ValueError):self.check((h,x.tobytes(),b,c))
  with self.assertRaises(ValueError):self.check((h[:-1],a,b,c))
 def test_real_four_frame_inherited_admission_and_closure(self):
  with tempfile.TemporaryDirectory()as t:
   parent=Path(t);root,rh,ref,fh=previous.Controls().fixture(parent);reference=json.loads(ref.read_text());report=json.loads((root/'report.json').read_text());h,a,b,c=self.fixture(128,16,48);values={'gdn_normalized_qkv':h,'gdn_decay_beta':a,'gdn_state_before':b,'gdn_state_after':c}
   for row in reference['current43_capture_after']['requests']:
    for f in row['raw33']:
     if f['name']in values:
      p=Path(f['path']);p.write_bytes(values[f['name']]);f['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();report['artifacts'][str(p.relative_to(root))]=f['sha256']
    meta=Path(row['metadata']);frame=json.loads(meta.read_text());frame['fields']=row['raw33'];meta.write_text(json.dumps(frame));row['metadata_sha256']=hashlib.sha256(meta.read_bytes()).hexdigest();report['artifacts'][str(meta.relative_to(root))]=row['metadata_sha256']
   (root/'report.json').write_text(json.dumps(report));rh=hashlib.sha256((root/'report.json').read_bytes()).hexdigest();reference['current43_capture_after']['public_report_sha256']=rh;reference['current43_capture_before']=copy.deepcopy(reference['current43_capture_after']);reference['expected_actor_report_sha256']=rh;ref.write_text(json.dumps(reference));fh=hashlib.sha256(ref.read_bytes()).hexdigest();result=q.run(root,rh,ref,fh,parent/'audit');self.assertTrue(result['audit']['all_four_states_exact']);self.assertEqual(len(result['consumed_evidence']),22);self.assertFalse(result['full_model_math_qualified'])
 def test_private_port_keeps_original_globals(self):
  self.assertNotIn('FIELDS',q.original.__dict__);self.assertNotEqual(q._PORTS['PLAN'],q.original.PLAN);self.assertIs(q.run.__globals__['audit'],q._PORTS['audit']);self.assertIs(q.run.__globals__['source_binding'].__globals__,q._PORTS)
if __name__=='__main__':unittest.main()
