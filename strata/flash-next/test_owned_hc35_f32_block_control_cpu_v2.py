"""Exact source F32 metadata admission; no weights, compiled helper or device."""
import copy,hashlib,json,struct,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import owned_hc35_f32_block_control_v2 as own

INVENTORY=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/current-gguf-inventory.json')
def metadata():
 rows=json.loads(INVENTORY.read_text())['files'];file=next(f for f in rows if '/UD-Q4_K_XL/' in f['path']);return copy.deepcopy(file['metadata']['qwen4exp.attention.layer_norm_rms_epsilon'])
class Provider:
 actual_source=True;source_identity_sha256='a'*64
 def __init__(self,row):
  self.reader=SimpleNamespace(files=[{'metadata':{'qwen4exp.attention.layer_norm_rms_epsilon':row}}]);self.source_contract={'original_rms_epsilon':row['value'],'effective_expert_scale':1.};self.payload_reads=0
 def rows(self,*a):self.payload_reads+=1;raise AssertionError('No model payload reads in metadata ctor')
class Controls(unittest.TestCase):
 def ctor(self,provider):
  with patch.object(own,'derive_schedule',return_value={}),patch.object(own,'source_binding',return_value={'CPU_SOURCE_METADATA_ONLY':True}),patch.object(own,'OriginalStorageProjector') as projector:
   return own.OwnedHcBlockControl(provider,'a'*64,[],{},'/CPU_ONLY',lambda:{'CPU_HOST_MOCK':True})
 def test_actual_saved_inventory_scalar_F32_ctor_accepts_without_payload(self):
  p=Provider(metadata());m=self.ctor(p);self.assertEqual(p.payload_reads,0);self.assertEqual(m.eps,float(np.float32(1e-6)));self.assertEqual(m.epsilon_binding['epsilon_LE_F32_hex'],'bd378635');self.assertNotEqual(m.eps,1e-6)
 def test_neighbor_and_F64_literal_rejected_before_source_and_rows(self):
  for value in [1e-6,float(np.nextafter(np.float32(1e-6),np.float32(np.inf))),float(np.nextafter(np.float32(1e-6),np.float32(0.)))]:
   row=metadata();row['value']=value;row['encoded_value_sha256']=hashlib.sha256(struct.pack('<f',value)).hexdigest();p=Provider(row)
   with patch.object(own,'derive_schedule',return_value={}),patch.object(own,'source_binding') as source:
    with self.assertRaises(ValueError):own.OwnedHcBlockControl(p,'a'*64,[],{},'/CPU_ONLY',lambda:{})
    source.assert_not_called();self.assertEqual(p.payload_reads,0)
 def test_scalar_metadata_type_shape_digest_negatives(self):
  changes=[{'type':12},{'type':True},{'value':[own.EPSILON]},{'value':np.asarray([own.EPSILON])},{'value':np.float32(own.EPSILON)},{'encoded_value_sha256':'0'*64}]
  for change in changes:
   row=metadata();row.update(change)
   with self.subTest(change=str(change)),self.assertRaises(ValueError):own.epsilon_contract(Provider(row))
 def test_exact_epsilon_addition_used_in_source_RMS(self):
  r=own.rms_argument([1.]*2560,own.EPSILON);self.assertEqual(r['rsqrt_argument_f32'],own.host.f32(1+own.EPSILON));self.assertEqual(r['square_sum_f32'],2560.)
 def test_V1_arithmetic_and_guard_functions_unchanged(self):
  import ast,owned_hc35_f32_block_control_v1 as old
  source=lambda module:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
  a,b=source(old),source(own)
  for name in ('raw','dot','rms_argument','rsqrt_candidate','sigmoid_candidate','mixed_candidates'):self.assertEqual(a[name],b[name])
 def test_V1_failure_and_source_preserved_by_new_plan(self):
  plan=json.loads(own.SOURCE_PLAN.read_text());self.assertEqual(plan['preserved_V1_failure']['reason'],'Original RMS/source contract changed');self.assertFalse(plan['actual_original_computation_performed_by_agent'])
if __name__=='__main__':unittest.main()
