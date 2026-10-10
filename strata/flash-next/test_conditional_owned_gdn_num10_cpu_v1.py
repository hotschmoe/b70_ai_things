"""CPU synthetic conditional mixer replay and admission controls; no payload."""
import ast,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
import conditional_owned_gdn_num10_v1 as q
from owned_layer0_gdn_replay_v1 import OwnedLayer0GdnReplay,physical_recurrent,physical_conv
from full48_synthetic_original_roles_v1 import SyntheticOriginalRoles
from test_full48_route_reference_cpu_v2 import config
import test_ple_prompt35_cpu_v1 as source35

class ConditionalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):source35.SourceTests.setUpClass();cls.source=source35.SourceTests.source
 @classmethod
 def tearDownClass(cls):source35.SourceTests.tearDownClass()
 def test_saved_own_mixer_exact_replay_and_state_ownership(self):
  args,env=config();model=OwnedLayer0GdnReplay(SyntheticOriginalRoles(),'a'*64,args,env,self.source);owned=model.tokens([19,20,21,22]);row=owned['rows'][-1];before=q.state_from_physical(row['fields_owned']['gdn_state_before'],row['fields_owned']['gdn_conv_before']);baseline={k:v.copy() for k,v in before.items()}
  fields,packets,events=q.mixer_case(model,row['fields_owned']['attn_mixed'],before,'verifier')
  for name,value in fields.items():self.assertTrue(np.array_equal(value,row['fields_owned'][name]),name)
  self.assertEqual(packets,row['packets_owned']);self.assertTrue(events)
  for name in before:self.assertTrue(np.array_equal(before[name],baseline[name]))
  conditional=row['fields_owned']['attn_mixed'].copy();conditional[0]+=1
  changed,_,_=q.mixer_case(model,conditional,before,'verifier');self.assertFalse(np.array_equal(changed['gdn_qkv'],fields['gdn_qkv']))
  for name in before:self.assertTrue(np.array_equal(before[name],baseline[name]))
 def test_physical_permutation_and_no_shared_state(self):
  recurrent=np.arange(128*48*128,dtype=np.float32).reshape(128,48,128);conv=np.arange(10240*3,dtype=np.float32).reshape(10240,3);state=q.state_from_physical(recurrent,conv)
  self.assertEqual(state['recurrent'][3,7,11],recurrent[7,3,11]);self.assertEqual(state['conv'][2,17],conv[17,2]);state['conv'][2,17]=-1;self.assertNotEqual(conv[17,2],-1)
  with self.assertRaises(ValueError):q.state_from_physical(recurrent.reshape(48,128,128),conv)
 def test_route_shape_nonfinite_rejected_before_mixer(self):
  class Model:pass
  for mixed,route in [(np.zeros(2560),'prefill'),(np.zeros(4),'verifier'),(np.full(2560,np.nan),'verifier')]:
   with self.assertRaises(ValueError):q.mixer_case(Model(),mixed,{},route)
 def test_completed_report_hash_fails_before_provider(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'report.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'report changed'):q.preflight(root,root,'a'*64,root/'missing',[])
 def test_new_lane_source_and_no_frozen_math_edit(self):
  text=Path(q.__file__).read_text();ast.parse(text)
  for marker in ('saved_own_input_and_state_replay','native_mixed_own_beforestate_CONDITIONAL','native_mixed_native_beforestate_CONDITIONAL','post_CPU_source4','post_binding==binding','tile_bytes=64<<20','derive_schedule','original_role_binding','numpy_binary_sha256'):self.assertIn(marker,text)
  self.assertNotIn('gdn.mixer =',text);self.assertNotIn('tokens(',text)
  self.assertIn('len(report[\'cases\'])==2+int(a.native_beforestate_case)',text)

if __name__=='__main__':unittest.main()
