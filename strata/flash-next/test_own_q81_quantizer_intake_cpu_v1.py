import copy,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
import own_q81_quantizer_intake_v1 as q
from independent_q8_1_activation_v1 import encode,NATIVE,OFFICIAL
class Controls(unittest.TestCase):
 def fixture(self,root):
  records=[]
  for p,token in enumerate(q.IDS):
   raw=np.arange(6144,dtype='<f4').tobytes();path=root/('own%d.f32'%p);path.write_bytes(raw);b={'path':str(path),'shape':[6144],'role':'blk.3.attn_output.weight.own_input','position':p,'source_identity_sha256':'a'*64,'actual_original_provider':True,'captured_operand_used':False,'snapshot_is_math_input':False,'encoding':'LE_F32','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
   records.append({'position':p,'token':token,'route':'verifier','predecessor_cells':p,'captured_inputs_or_states_used':False,'projections':[{'role':'blk.3.attn_output.weight','own_input_snapshot':b}]})
  return {'records':records}
 def run_case(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);own=self.fixture(root)
   if kind=='token':own['records'][2]['token']=q.IDS[1]
   if kind=='position':own['records'][2]['position']=1
   if kind=='layer':own['records'][0]['projections'][0]['role']='blk.4.attn_output.weight'
   if kind=='state':own['records'][3]['predecessor_cells']=2
   if kind=='captured':own['records'][0]['captured_inputs_or_states_used']=True
   if kind=='mutation':(root/'own0.f32').write_bytes(bytes(6144*4))
   if kind=='valid':
    raw,rows=q.extract(root,own,'a'*64,'output_gated');self.assertEqual(len(raw),4*6144*4);self.assertEqual(len(rows),4)
   else:
    with self.assertRaises(ValueError):q.extract(root,own,'a'*64,'output_gated')
 def test_valid(self):self.run_case('valid')
 def test_wrong_token(self):self.run_case('token')
 def test_wrong_position(self):self.run_case('position')
 def test_wrong_layer(self):self.run_case('layer')
 def test_wrong_predecessor(self):self.run_case('state')
 def test_captured_inputs(self):self.run_case('captured')
 def test_raw_mutation(self):self.run_case('mutation')
 def test_half_away_boundary_and_reciprocal_rival(self):
  x=np.zeros(32,dtype='<f4');x[0]=np.float32(31.6876964569);x[1]=np.float32(3.8673961163)
  self.assertNotEqual(encode(x.tobytes(),1,32,NATIVE),encode(x.tobytes(),1,32,OFFICIAL))
  x[0]=127;x[1]=-6.5;packet=encode(x.tobytes(),1,32);self.assertEqual(packet[5],249)
if __name__=='__main__':unittest.main()
