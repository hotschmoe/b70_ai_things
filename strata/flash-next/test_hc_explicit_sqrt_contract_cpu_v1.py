import subprocess,tempfile,unittest
from pathlib import Path
import prepare_hc_explicit_sqrt_contract_v1 as p
class Controls(unittest.TestCase):
 def test_actual_source_apply_reverse_and_preservation(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);result=p.prepare(root/'prepared');target=root/'tree/sycl/src/kernels/hc_native_composition.cpp';target.parent.mkdir(parents=True);target.write_bytes(p.BASE.read_bytes());patch=root/'prepared/hc-explicit-sqrt-divide.patch';subprocess.run(['git','apply',str(patch)],cwd=root/'tree',check=True,capture_output=True);self.assertEqual(p.sha(target),result['result_sha256']);subprocess.run(['git','apply','-R',str(patch)],cwd=root/'tree',check=True,capture_output=True);self.assertEqual(p.sha(target),p.BASE_SHA)
 def test_guard_and_both_exact_expressions(self):
  text=p.transform(p.BASE.read_text());self.assertLess(text.index('if(explicit_sqrt_divide<0)'),text.index('strata::q_of(stream)'));self.assertEqual(text.count('1.0f/sycl::sqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps)'),1);self.assertEqual(text.count('sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps)'),1);self.assertEqual(text.count('hc_prefill_projection_tile()'),p.BASE.read_text().count('hc_prefill_projection_tile()'));self.assertEqual(text.count('hc_q8_0_project_f32_token_reuse'),2)
if __name__=='__main__':unittest.main()
