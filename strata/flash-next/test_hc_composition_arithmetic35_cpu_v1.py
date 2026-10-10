"""Synthetic CPU/source controls only; no compile, helper execution or device."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import hc_composition_arithmetic35_fixture_v1 as fixture
import prepare_hc_composition35_leaf_v1 as producer

class Controls(unittest.TestCase):
 def test_actual_canonical_18_fields_counts_independently(self):
  self.assertEqual(len(fixture.FIELDS),19);perrow=5*4+6*10240+4*320+3*2560+2;self.assertEqual(perrow,70422)
  self.assertEqual(fixture.corpus_counts(),{'cases':4,'frames':36,'output_floats':(1+2+1+2)*3*3*perrow,'fields_per_frame':19,'negative_cases':2});self.assertEqual(producer.corpus_contract()['words'],3802788)
 def test_wrong_count_failclosed_before_host_boundary(self):
  p={'computed_corpus_counts':dict(fixture.corpus_counts(),output_floats=3802681)}
  with self.assertRaises(ValueError):fixture.count_binding(p)
 def test_real_synthetic_corpus_extents_and_signed_half_scales(self):
  for case in fixture.fixtures():
   d=fixture.case_data(case);self.assertEqual(len(d['residual.f32']),case['t']*10240*4);self.assertEqual(len(d['down.q8_0']),320*10240//32*34);self.assertEqual(len(d['up.q8_0']),10240*320//32*34)
   codes=np.frombuffer(d['down.q8_0'],dtype=np.uint8).reshape(-1,34)
   if case['seed']:self.assertEqual(codes[0,:2].tobytes(),np.asarray([2**-24],dtype='<f2').tobytes());self.assertTrue((codes[:,2:].view(np.int8)<0).any())
 def test_unknown_seed_shape_or_roster_refused(self):
  for case in [{'id':'analytic_t1','t':3,'seed':0},{'id':'heldout_t1','t':1,'seed':178}]:
   with self.assertRaises(ValueError):fixture.case_data(case)
 def test_actual_and_shadow_provenance_are_disjoint(self):
  self.assertEqual(len(fixture.ACTUAL),8);self.assertNotIn('norm_square_sum_shadow',fixture.ACTUAL);self.assertNotIn('pre_silu_secondary_projection',fixture.ACTUAL);self.assertNotIn('exp_gate_shadow',fixture.ACTUAL)
 def test_relation_mutations_detected_without_candidate_selection(self):
  fields={k:b'CPU_SYNTHETIC' for k in fixture.FIELDS};self.assertTrue(all(fixture.relationships(fields,True).values()))
  bad=dict(fields,rs=b'CHANGED');self.assertFalse(fixture.relationships(bad,True)['actual_rs_equals_shadow']);bad=dict(fields,standalone_write=b'CHANGED');self.assertFalse(fixture.relationships(bad,True)['pending_equals_standalone']);self.assertTrue(fixture.relationships(bad,False)['pending_equals_standalone'])
 def test_canonical_export_and_rehash_cannot_admit_changed_synthetic_weights(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'CPU_SOURCE_PLAN.json';source.write_text('{}');out=root/'package'
   with patch.object(fixture,'SOURCE_PLAN',source),patch.object(fixture,'source_binding',return_value={'CPU_ONLY':True}),patch.object(fixture,'host_binding',return_value={'CPU_MOCK_HOST':True}):
    fixture.prepare(out);fixture.input_binding(out);path=out/'inputs/analytic_t1.down.q8_0';raw=bytearray(path.read_bytes());raw[2]=1;path.write_bytes(raw);manifest=json.loads((out/'manifest.json').read_text());manifest['cases'][0]['inputs']['down.q8_0']['sha256']=fixture.sha(path);fixture.write(out/'manifest.json',manifest)
    with self.assertRaises(ValueError):fixture.input_binding(out)
 def test_graph_owner_and_alias_reset_source_contract(self):
  src=(Path(fixture.__file__).parent/'hc_composition_arithmetic35_gpu_v1.cpp').read_text();self.assertIn('GraphOwner graphs(q);',src);self.assertIn('graphs.executable',src);self.assertIn('q.ext_oneapi_graph(*graphs.executable).wait_and_throw()',src);self.assertIn('q.memcpy(R,host_input.data(),host_input.size())',src);self.assertIn('apply==2?R:R_out',src);self.assertLess(src.index('GraphOwner graphs(q);'),src.index('mem.release();'))
 def test_device_intrinsic_namespace_and_candidate_not_policy(self):
  src=(Path(fixture.__file__).parent/'hc_composition_arithmetic35_gpu_v1.cpp').read_text();self.assertIn('sycl::exp(-a.gate[i])',src);self.assertIn('sycl::rsqrt(arg)',src);self.assertNotIn('sycl::native::exp',src);self.assertIn('normal_model_graph_qualified=0',src)
 def test_runtime_purpose_no_original_payload_or_hidden_outputs(self):
  src=(Path(fixture.__file__).parent/'hc_composition_arithmetic35_gpu_v1.cpp').read_text();self.assertNotIn('.gguf',src);self.assertNotIn('/expected',src);self.assertIn('actual HC graph absent',src);self.assertIn('single_fp_config',src)
 def test_FP_shadow_real_USM_operands_and_no_policy_gate(self):
  data=fixture.case_data(fixture.fixtures()[0]);self.assertEqual(np.frombuffer(data['fp_control.f32'],dtype='<f4').tolist(),[2**-126,.5,2**-149,2**24]);src=(Path(fixture.__file__).parent/'hc_composition_arithmetic35_gpu_v1.cpp').read_text();self.assertIn('fp_inputs[t*4+j*2]',src)
if __name__=='__main__':unittest.main()
