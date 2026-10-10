"""PLE phase crossing and exact own HC operand publication, synthetic CPU only."""
import copy,inspect,tempfile,unittest
from pathlib import Path
import numpy as np
import full48_owned_hc_device_rs_v3 as reference
import owned_hc_device_rs_protocol_v1 as protocol
import owned_hc_device_rs_experiment_v3 as experiment
import test_owned_hc_device_rs_cpu_v1 as original_tests
import qualify_owned_hc_device_rs_v3 as q
class Tests(unittest.TestCase):
 def model(self,root):
  model=object.__new__(reference.OwnedHcDeviceRsReference);model.operand_root=root/'rs-operands';model.operand_root.mkdir();model.device_rs=protocol.OwnDeviceRsClient(original_tests.reply,original_tests.qualification());model.device_rs.records=[{'synthetic_prior_ordinal':i} for i in range(3)];model.owned_rms_records=[];model.epsilon_contract={'epsilon':reference.hc.EPSILON};return model
 def test_layer1_PLE_postprojection_operand_not_pre_input_phase(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);model=self.model(root);before=np.ones((4,2560),dtype='<f4');owned_PLE_delta=np.full((4,2560),.25,dtype='<f4');after=np.asarray(before+owned_PLE_delta,dtype='<f4');model.resolve_owned_rs(after,'blk.1.hc_attn_');row=model.owned_rms_records[0];device=model.device_rs.records[-1];saved=experiment.bind_own_operand(row,device,root);self.assertEqual(saved.tobytes(),after.tobytes());self.assertNotEqual(saved.tobytes(),before.tobytes());self.assertEqual(row['device_sequence'],4);self.assertEqual(row['source_FMA_XOR_arguments'],[reference.hc.rms_argument(v,reference.hc.EPSILON) for v in after]);self.assertNotEqual(row['source_FMA_XOR_arguments'],[reference.hc.rms_argument(v,reference.hc.EPSILON) for v in before])
 def test_exact_operand_role_ordinal_path_hash_and_shape_controls(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);model=self.model(root);model.resolve_owned_rs(np.ones((4,2560),dtype='<f4'),'blk.1.hc_attn_');row=model.owned_rms_records[0];device=model.device_rs.records[-1]
   for kind in ('role','ordinal','path','sha','shape','capture'):
    bad=copy.deepcopy(row);b=bad['owned_operand']
    if kind=='role':b['role']='blk.0.hc_attn_'
    elif kind=='ordinal':b['sequence']=100
    elif kind=='path':b['path']=str(root/'wrong.f32')
    elif kind=='sha':b['sha256']='WRONG'
    elif kind=='shape':b['shape']=[10240]
    else:b['captured_operand_used']=True
    with self.assertRaises(ValueError):experiment.bind_own_operand(bad,device,root)
 def test_operand_snapshot_is_output_not_an_arithmetic_input(self):
  s=inspect.getsource(reference.OwnedHcDeviceRsReference.resolve_owned_rs);self.assertIn('super().resolve_owned_rs(matrix,stem)',s);self.assertNotIn('read_bytes',s);self.assertLess(s.index("path.open('xb')"),s.index('super().resolve_owned_rs'))
 def test_snapshot_cannot_overwrite_previous_owned_evidence(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);model=self.model(root);model.resolve_owned_rs(np.ones((4,2560),dtype='<f4'),'blk.1.hc_attn_');model.device_rs.records.pop()
   with self.assertRaises(FileExistsError):model.resolve_owned_rs(np.ones((4,2560),dtype='<f4'),'blk.1.hc_attn_')
 def test_arithmetic_methodset_inherited_unchanged(self):
  self.assertEqual({k for k,v in reference.OwnedHcDeviceRsReference.__dict__.items() if callable(v)},{'__init__','resolve_owned_rs'});self.assertIs(reference.OwnedHcDeviceRsReference.hc_read,reference.original.OwnedHcDeviceRsReference.hc_read);self.assertIs(reference.OwnedHcDeviceRsReference.tokens,reference.original.OwnedHcDeviceRsReference.tokens)
 def test_new_consumer_selects_new_work_exact_operand_mapping(self):
  s=Path(q.__file__).read_text();self.assertIn('owned_hc_device_rs_experiment_v3',s);self.assertIn('owned_hc_device_rs_work_v3',s);s=Path(experiment.__file__).read_text();self.assertNotIn("arrays'][f'p{position}_l{layer}_{phase_name}']",s);self.assertIn('bind_own_operand(row,device,work_root)',s)
 def test_failed_executed_V2_source_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('owned-hc-device-rs-source-plan-v2.json')),'de8dd54595eb8d15719680ccecc79e8bc7adc256c6902effcecdb84f09e04444')
if __name__=='__main__':unittest.main()
