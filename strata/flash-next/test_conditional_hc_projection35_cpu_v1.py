#!/usr/bin/env python3
"""Synthetic conditional HC source/dtype/roster controls. No host ELF or model run."""
import copy,hashlib,json,struct,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import conditional_hc_projection35_v1 as q

class PackedSynthetic:
 actual_source=False
 def __init__(self,root):
  self.root=Path(root);self.reader=SimpleNamespace(tensors={});self.calls=[]
  for case in q.ROSTER.values():
   cols,total=case['shape'];row=(struct.pack('<e',.125)+bytes([1]*32))*(cols//32) if case['kind']=='Q8_0' else np.full(cols,.125,dtype='<f4').tobytes();path=self.root/(case['role']+'.raw');path.write_bytes(row*total);st=path.stat();sig=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns];file={'path':str(path),'tensor_data_offset':0,'size_bytes':st.st_size};tensor={'type':case['kind'],'shape_ggml_order':list(case['shape']),'absolute_offset':0};self.reader.tensors[case['role']]=(file,tensor,sig)
 def rows(self,name,indices):
  self.calls.append((name,list(indices)));cols=q.ROSTER[next(key for key,v in q.ROSTER.items() if v['role']==name)]['shape'][0];return np.full((len(indices),cols),.125)

def fake_runner(helper,helper_sha,op,a,b):
 # Own tiny synthetic lane: all weights0.125, allinputs1, exactly representable.
 count=len(b)//4;raw=struct.pack('<f',count*.125);proof={'helper_sha256':helper_sha,'input_sha256':[hashlib.sha256(x).hexdigest() for x in (a,b)],'output_sha256':hashlib.sha256(raw).hexdigest(),'host_gradual_input_output_probes_passed':True,'implementation_arithmetic_qualified':False,'device_intrinsics_qualified':False,'model_math_qualified':False,'tolerance_gate':None};return raw,proof

def values():return {'attn_hc_low':np.ones(320),'attn_hc_normalized':np.ones((4,2560)),'attn_hc_gate':np.full((4,2560),40.),'attn_hc_inject':np.full(4,1280.)}

class ConditionalTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='HC35-conditional-CPU-');self.root=Path(self.tmp.name);self.p=PackedSynthetic(self.root)
 def tearDown(self):self.tmp.cleanup()
 def run_case(self,include_down=False,runner=fake_runner,targets=None):return q.selected_projection(self.p,values() if targets is None else targets,'CPU_NOT_EXECUTED_ELF','a'*64,include_down,runner)
 def test_exact_predeclared_rows_and_inputs(self):
  cases=q.selected_cases();self.assertEqual(cases['up']['rows'],[0,1,1123,10239]);self.assertEqual(cases['inject']['rows'],[0,1,2,3]);self.assertNotIn('down',cases);self.assertEqual(q.selected_cases(True)['down']['rows'],[0,1,159,319]);self.assertEqual(cases['up']['input'],'attn_hc_low');self.assertEqual(cases['inject']['input'],'attn_hc_normalized')
 def test_selected_projection_dual_lanes_and_raw_weights(self):
  results,weights=self.run_case();self.assertEqual(len(weights),8)
  for name,row in results.items():
   self.assertEqual(row['source32_F32_output'].dtype,np.dtype('<f4'));self.assertEqual(row['mathematical_F64_output'].dtype,np.dtype('<f8'));self.assertTrue(row['captured_inputs_used']);self.assertFalse(row['full_model_math_qualified']);self.assertFalse(row['numeric_pass_claim']);self.assertTrue(row['native_output_observed']);self.assertTrue(all(c['F32_bytes_equal'] for c in row['comparisons'].values()));self.assertTrue(all(c['tolerance_gate'] is None for c in row['comparisons'].values()))
 def test_down_preSiLU_no_postSiLU_comparison(self):
  result,_=self.run_case(True);down=result['down'];self.assertIsNone(down['selected_native_output']);self.assertFalse(down['native_output_observed']);self.assertEqual(down['comparisons'],{});self.assertTrue(down['down_preSiLU_vs_postSiLU_comparison_forbidden']);self.assertEqual(down['input_field'],'attn_hc_normalized')
 def test_f64_lane_is_not_the_F32_output_recast(self):
  def changed(*args):
   raw,receipt=fake_runner(*args);raw=struct.pack('<f',struct.unpack('<f',raw)[0]+.25);receipt['output_sha256']=hashlib.sha256(raw).hexdigest();return raw,receipt
  rows,_=self.run_case(runner=changed)
  self.assertTrue(np.all(rows['up']['mathematical_F64_output']==40.));self.assertTrue(np.all(rows['up']['source32_F32_output']==40.25));self.assertFalse(rows['up']['comparisons']['source32_FMA_XOR_F32']['F32_bytes_equal']);self.assertTrue(rows['up']['comparisons']['original_sameinput_F64']['F32_bytes_equal'])
 def test_actualcaptured_operands_not_modified_by_diagnostic(self):
  target=values();before=target['attn_hc_normalized'].tobytes();self.run_case(targets=target);self.assertEqual(target['attn_hc_normalized'].tobytes(),before)
 def test_helper_wrong_dtype_extent_nonfinite_rejected(self):
  for raw in (struct.pack('<d',1),b'',struct.pack('<f',float('nan'))):
   def invalid(*args):
    _,receipt=fake_runner(*args);receipt['output_sha256']=hashlib.sha256(raw).hexdigest();return raw,receipt
   with self.assertRaises(ValueError):self.run_case(runner=invalid)
 def test_helper_input_output_source_scope_receipt_rejected(self):
  for key,value in [('helper_sha256','b'*64),('input_sha256',['c'*64]*2),('output_sha256','d'*64),('host_gradual_input_output_probes_passed',False),('model_math_qualified',True),('tolerance_gate',1e-6)]:
   def wrong(*args):raw,receipt=fake_runner(*args);receipt[key]=value;return raw,receipt
   with self.assertRaises(ValueError):self.run_case(runner=wrong)
 def test_native_operand_nonfinite_wrongshape_notexactF32(self):
  for field,value in [('attn_hc_low',np.ones(319)),('attn_hc_low',np.full(320,np.nan)),('attn_hc_low',np.full(320,1.+2**-40))]:
   target=values();target[field]=value
   with self.assertRaises(ValueError):self.run_case(targets=target)
 def test_native_comparison_output_shape_nonfinite_rejected(self):
  for field,value in [('attn_hc_gate',np.ones(10239)),('attn_hc_gate',np.full(10240,np.nan)),('attn_hc_inject',np.ones(3))]:
   target=values();target[field]=value
   with self.assertRaises(ValueError):self.run_case(targets=target)
 def test_original_role_wrongdtype_shape_rejected(self):
  _,tensor,_=self.p.reader.tensors[q.ROSTER['up']['role']];tensor['type']='Q4_K'
  with self.assertRaises(ValueError):self.run_case()
  tensor['type']='Q8_0';tensor['shape_ggml_order']=[320,10239]
  with self.assertRaises(ValueError):self.run_case()
 def test_packed_original_row_source_stat_decode_and_bounds_negative(self):
  case=q.ROSTER['up'];raw,decoded,binding=q.packed_row(self.p,case,1123);self.assertEqual(len(raw),340);self.assertTrue(np.all(decoded==.125));self.assertEqual(binding['row'],1123)
  with self.assertRaises(ValueError):q.packed_row(self.p,case,1122)
  path=Path(self.p.reader.tensors[case['role']][0]['path']);path.write_bytes(path.read_bytes()+b'X')
  with self.assertRaises(ValueError):q.packed_row(self.p,case,1123)
 def test_decoder_disagreement_rejected(self):
  with patch.object(self.p,'rows',lambda name,idx:np.zeros((len(idx),320))):
   with self.assertRaises(ValueError):self.run_case()
 def test_final_helper_compile_analytic_adapter_runtime_metadata_admission(self):
  import qualify_hc35_host_runtime_v1 as runtime
  marker={'CPU_METADATA_ONLY':True,'report_sha256':q.HOST_RUNTIME_REPORT_SHA,'tiny_known_answer_cases':15,'actual_adapter_cases':3}
  with patch.object(runtime,'finalized_binding',lambda *args:marker):
   proof=q.host_admission(q.HOST_ROOT/'hc35-host',q.HOST_ROOT,q.HOST_RUNTIME_ROOT);self.assertEqual(proof['current_host_runtime_finalized_binding'],marker);self.assertFalse(proof['full_model_math_qualified']);self.assertIsNone(proof['tolerance_gate'])
   realsha=q.sha
   def badsha(path):return 'f'*64 if Path(path).name=='v2-final-adapter-receipt.json' else realsha(path)
   with patch.object(q,'sha',badsha):
    with self.assertRaises(ValueError):q.host_admission(q.HOST_ROOT/'hc35-host',q.HOST_ROOT,q.HOST_RUNTIME_ROOT)
   with self.assertRaises(ValueError):q.host_admission(self.root/'WRONG_ELF',q.HOST_ROOT,q.HOST_RUNTIME_ROOT)
   realread=q.read
   def badread(path):
    value=copy.deepcopy(realread(path))
    if Path(path).name=='analytic-receipt.json':value['rows']=value['rows'][:-1]
    return value
   with patch.object(q,'read',badread):
    with self.assertRaises(ValueError):q.host_admission(q.HOST_ROOT/'hc35-host',q.HOST_ROOT,q.HOST_RUNTIME_ROOT)
 def test_preserve_raw_output_and_encoding(self):
  item=q.save_raw(self.root,'fresh.f64',struct.pack('<d',40.),'LE_F64',[1]);self.assertEqual(item['bytes'],8);self.assertEqual(item['encoding'],'LE_F64');self.assertEqual(item['sha256'],q.sha(self.root/'fresh.f64'))
  with self.assertRaises(ValueError):q.save_raw(self.root,'fresh.f64',b'X','LE_F32',[1])
if __name__=='__main__':unittest.main()
