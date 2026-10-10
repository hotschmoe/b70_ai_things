"""Synthetic corpus/hostreceipt/raw collector tests; no helper/GPU/Docker/compile."""
import copy,hashlib,json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import hc_projection_arithmetic35_fixture_v1 as q
import hc_f32_arithmetic35_host_v2 as a

class CorpusTests(unittest.TestCase):
 def test_analytic_knownanswers_and_effective_negatives(self):
  cases={case['id']:case for case in q.fixtures()}
  expected={'xor_order':[1.],'xor_order_negative':[0.],'lane_cancel':[0.],'fused_cancel':[-2**-46],'round_midpoints':[1.,1+2**-22],'q8_signed':[-128.,127.],'q8_weight_negative':[-127.,127.],'q8_half_subnormal':[1.],'f32_subnormal':[2**-149]}
  for label,want in expected.items():
   case=cases[label];rowbytes=case['k']*4 if case['type']=='F32' else case['k']//32*34;got=[]
   for row in range(case['m']):
    weights=case['weights'][row*rowbytes:(row+1)*rowbytes];raw=a.dot32_f32(weights,case['input']) if case['type']=='F32' else a.q8dot32_f32(weights,case['input']);got+=a.words(raw)
   self.assertEqual(got,want,label)
 def test_fixed_heldout_seed_shapes_and_complete_T2_outputs(self):
  first=q.fixtures();self.assertEqual(first,q.fixtures());held=[row for row in first if row['id'].startswith('heldout_')];self.assertEqual(len(held),4)
  for row in held:
   self.assertEqual((row['m'],row['t']),(3,2));self.assertIn(row['k'],(320,10240));self.assertEqual(len(row['input']),row['k']*8);self.assertEqual(len(row['weights']),row['m']*(row['k']*4 if row['type']=='F32' else row['k']//32*34))
 def test_no_original_reference_or_quantizer_modification(self):
  text=Path(q.__file__).read_text();self.assertNotIn('Full48OwnedComposition',text);self.assertNotIn('model_shards',text);self.assertNotIn('tolerance =',text)

class RawCollectorTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.inputs=self.root/'inputs';self.raw=self.root/'raw';self.raw.mkdir();self.log=self.root/'log';self.host={'build_binding':{'helper':{'sha256':'CPU_mock_only'}},'model_math_qualified':False}
  def helper(path,digest,op,w,x):
   raw=a.dot32_f32(w,x) if op=='dot' else a.q8dot32_f32(w,x)
   return raw,{'helper_sha256':'CPU_mock_only','input_sha256':[hashlib.sha256(v).hexdigest() for v in (w,x)],'output_sha256':hashlib.sha256(raw).hexdigest(),'host_gradual_input_output_probes_passed':True,'model_math_qualified':False}
  self.bind=patch.object(q,'host_binding',return_value=self.host);self.bind.start();self.addCleanup(self.bind.stop)
  with patch.object(a,'run_helper',side_effect=helper),patch.object(a,'sha',return_value='CPU_mock_only'):
   # Mock real helper execution only; synthetic raw/hash/corpus production actual.
   with patch.object(q,'source_binding',return_value={}):self.manifest=q.prepare(self.inputs,self.root/'hostrun',self.root/'helper',self.root/'build')
  self.lines=['HC35_PROJECTION_CONFIG synthetic=1 device=CPU mock queue=0x1 ze_context=0x2 ze_device=0x3 in_order=1 device_intrinsics_qualified=0 model_math_qualified=0','HC35_PROJECTION_DEVICE backend=level_zero vendor=CPU mock driver=mock affinity=0 selector=level_zero:gpu physical_card_parent_mapping_required=1']
  for row in self.manifest['cases']:
   expected=Path(row['files']['expected']['path']).read_bytes();(self.raw/(row['id']+'.gpu.f32')).write_bytes(expected);self.lines.append('HC35_PROJECTION_ROW case=%s type=%s k=%d m=%d t=%d bytes=%d'%(row['id'],row['type'],row['k'],row['m'],row['t'],len(expected)))
  self.lines.append('HC35_PROJECTION_RESULT cases=%d execution_completed=1 all_owned_allocations_freed=1 numerical_comparison_done=0 device_intrinsics_qualified=0 model_math_qualified=0'%len(self.manifest['cases']));self.log.write_text('\n'.join(self.lines)+'\n')
 def collect(self):return q.collect(self.inputs,self.raw,self.log)
 def test_complete_mock_raw_pass_is_not_GPU_lifecycle_qualification(self):
  result=self.collect();self.assertTrue(result['numeric_bitwise_passed']);self.assertEqual(len(result['cases']),13);self.assertTrue(all(v['different_from_control'] for v in result['negative_controls']));self.assertFalse(result['actual_GPU_execution_qualified']);self.assertFalse(result['device_intrinsics_qualified']);self.assertFalse(result['model_math_qualified']);self.assertFalse(result['physical_device_identity_independently_qualified'])
 def test_subnormal_flush_is_failure_not_relaxed_tolerance(self):
  (self.raw/'f32_subnormal.gpu.f32').write_bytes(bytes(4));result=self.collect();self.assertFalse(result['numeric_bitwise_passed']);row=next(r for r in result['cases'] if r['id']=='f32_subnormal');self.assertEqual(row['differing_words'],1);self.assertIsNone(result['tolerance_gate']);self.assertFalse(result['device_intrinsics_qualified'])
 def test_wrong_weight_or_ineffective_negative_fails(self):
  path=self.raw/'q8_weight_negative.gpu.f32';path.write_bytes((self.raw/'q8_signed.gpu.f32').read_bytes());self.assertFalse(self.collect()['numeric_bitwise_passed'])
 def test_expectedrow_receipt_hash_cannot_be_invented(self):
  manifest=copy.deepcopy(self.manifest);manifest['cases'][0]['host_row_proofs'][0]['output_sha256']='f'*64;(self.inputs/'manifest.json').write_text(json.dumps(manifest))
  with self.assertRaises(ValueError):self.collect()
 def test_expectedbytes_and_updated_receipt_cannot_fake_source_schedule(self):
  manifest=copy.deepcopy(self.manifest);row=manifest['cases'][0];field=row['files']['expected'];Path(field['path']).write_bytes(struct.pack('<f',99.));field['sha256']=q.sha(field['path']);row['host_row_proofs'][0]['output_sha256']=field['sha256'];(self.inputs/'manifest.json').write_text(json.dumps(manifest))
  with self.assertRaisesRegex(ValueError,'independently recollected'):self.collect()
 def test_fresh_host_admission_or_library_drift_failclosed(self):
  with patch.object(q,'host_binding',return_value={'changed':True}):
   with self.assertRaises(ValueError):self.collect()
 def test_non_synthetic_operands_rejected_even_with_updated_input_hash(self):
  manifest=copy.deepcopy(self.manifest);field=manifest['cases'][0]['files']['input'];path=Path(field['path']);raw=bytearray(path.read_bytes());raw[:4]=struct.pack('<f',9.);path.write_bytes(raw);field['sha256']=q.sha(path);(self.inputs/'manifest.json').write_text(json.dumps(manifest))
  with self.assertRaises(ValueError):self.collect()
 def test_duplicate_producer_row_and_missing_or_truncated_raw_refused(self):
  self.log.write_text('\n'.join(self.lines+[self.lines[2]])+'\n')
  with self.assertRaises(ValueError):self.collect()
  self.log.write_text('\n'.join(self.lines)+'\n');path=self.raw/'heldout_f32_320.gpu.f32';path.write_bytes(path.read_bytes()[:-4])
  with self.assertRaises(ValueError):self.collect()
 def test_frozen_sourceplan_closure(self):self.assertEqual(q.source_binding()['corpus_cases'],13)

if __name__=='__main__':unittest.main()
