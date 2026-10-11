"""Source/CPU candidate controls; no compiler/device/model payload work."""
import hashlib,json,unittest
from pathlib import Path
import prepare_indexer_materialized41_candidate_v2 as p
class Controls(unittest.TestCase):
 def test_exact_actual_source_and_reconstructible_patch(self):
  before,after=p.reconstruct();self.assertEqual(p.PATCH.read_bytes(),p.patch_bytes());self.assertEqual(set(before),set(p.PATHS));self.assertTrue(all(before[n]!=after[n]for n in before));plan=p.build_proposal();self.assertEqual(plan['patches'][:-1],json.loads(p.BASE_PLAN.read_text())['patches']);self.assertEqual(plan['source41_materialized_half']['source40_input_bindings'],{n:hashlib.sha256(s.encode()).hexdigest()for n,s in before.items()})
 def test_all_eight_callers_keep_exact_original_nonraw_arguments(self):
  before,after=p.reconstruct();count=0
  for path in ('sycl/src/core/layer.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp'):
   old=p.calls(before[path]);new=p.calls(after[path]);self.assertEqual(len(old),len(new))
   for (_,_,name,a),(_,_,newname,b)in zip(old,new):
    self.assertEqual(name,newname);self.assertEqual(a[1:],b[1:]);self.assertTrue(b[0].startswith('strata::kernels::native_qsa_indexer_materialize_input('+a[0]+','));count+=1
  self.assertEqual(count,8)
 def test_commit_capacity_offsets_and_rejected_rows_do_not_read_raw(self):
  _,after=p.reconstruct();v=after['sycl/src/core/verify.cpp'];self.assertIn('idx_half_store_L_, idx_expanded_L_, (size_t) qsa_index * MT * ID',v);self.assertIn('idx_half_store_L_, idx_expanded_L_, (size_t) (qsa_index * MT + u) * ID',v);self.assertIn('b.take<uint16_t>(nQ * T * ID)',v)
  code=p.KERNEL;self.assertLess(code.index('if (pos>=0 && pos<max_cells)'),code.index('raw[i]'));self.assertIn('else half[i]=sycl::half(0.0f)',code)
 def test_two_materialized_kernels_before_unmodified_production_functions(self):
  before,after=p.reconstruct();path='sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp';self.assertEqual(after[path].replace(p.KERNEL,'').replace('#include <cmath>\n#include <cstdlib>\n#include <cstring>','#include <cmath>'),before[path]);self.assertLess(p.KERNEL.index('Indexer41MaterializedStore'),p.KERNEL.index('Indexer41MaterializedLoad'));self.assertIn('queue::in_order',p.KERNEL);self.assertIn('overlaps(spans[a],spans[b])',p.KERNEL)
 def test_OFF_no_new_device_work_or_scratch_access(self):
  body=p.KERNEL[p.KERNEL.index('const float* native_qsa_indexer_materialize_input'):];self.assertLess(body.index('if (!native_qsa_indexer_materialized_half_enabled()) return raw;'),body.index('validate(span)'));self.assertLess(body.index('return raw;'),body.index('queue->parallel_for'));self.assertIn('if (!value || std::strcmp(value, "0") == 0) return false',p.KERNEL);self.assertIn('requested && !native_qsa_indexer_enabled()',p.KERNEL)
 def test_prefill_count_and_carve_sequences_and_layer_capacity(self):
  _,after=p.reconstruct();f=after['sycl/src/prefill/prefill.cpp'];count=f[f.index('uint64_t qsa_set_bytes'):f.index('// Step 2b:')];self.assertLess(count.index('a.take<float>(T * 128, ok)'),count.index('a.take<uint16_t>(T * 128, ok)'));self.assertEqual(count.count('a.take<float>(T * 128, ok)'),2);self.assertIn('m.idx_half_store = b.take<uint16_t>(T * 128, ok)',f);self.assertIn('m.idx_expanded = b.take<float>(T * 128, ok)',f);self.assertIn('g.idx_key_dim * 6 + 32',after['sycl/src/core/layer.cpp'])
 def test_ON_identity_key_only_and_mutation_negative(self):
  before,after=p.reconstruct();f=after['sycl/src/program/generate.cpp'];key='c.switches.emplace_back("native_qsa_indexer_materialized_half", int64_t(1))';self.assertEqual(f.count(key),1);self.assertIn('if (strata::kernels::native_qsa_indexer_materialized_half_enabled())\n                    '+key,f);self.assertNotIn(key,before['sycl/src/program/generate.cpp']);self.assertRaises(ValueError,p.once,'same same','same','new');self.assertRaises(ValueError,p.replace_calls,'native_qsa_indexer_append(a,b,c,d,e,f,g,h,i,j);','verify')
if __name__=='__main__':unittest.main()

class OffLayoutControls(unittest.TestCase):
 def test_all_callers_pass_bases_not_null_pointer_arithmetic(self):
  _,after=p.reconstruct()
  for path in('sycl/src/core/layer.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp'):
   for _,_,_,args in p.calls(after[path]):
    helper=p.split_args(args[0][args[0].index('(')+1:-1]);self.assertNotIn('+',helper[7]);self.assertNotIn('+',helper[8]);self.assertEqual(len(helper),12)
  body=p.KERNEL[p.KERNEL.index('const float* native_qsa_indexer_materialize_input'):];self.assertLess(body.index('return raw;'),body.index('half_base + scratch_offset'));self.assertLess(body.index('count > scratch_capacity - scratch_offset'),body.index('half_base + scratch_offset'));self.assertIn('for (const auto& span : owners) validate(span)',body)
 def test_both_Verifier_public_headers_mirror_new_owner_fields(self):
  _,after=p.reconstruct()
  for path in('include/strata/core/verify.hpp','sycl/include/strata/core/verify.hpp'):
   self.assertEqual(after[path].count('uint16_t* idx_half_store_L_ = nullptr;'),1);self.assertEqual(after[path].count('float* idx_expanded_L_ = nullptr;'),1)
