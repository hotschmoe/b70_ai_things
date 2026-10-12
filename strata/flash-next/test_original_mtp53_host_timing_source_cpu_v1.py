import unittest
from pathlib import Path
import prepare_original_mtp53_host_timing_v1 as p
class Controls(unittest.TestCase):
 def test_actual_stage_source_no_extra_GPU_work(self):
  source=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/sycl/src/core/verify.cpp').read_text();out=p.target_source(source)
  for name in('ext_oneapi_graph','wait()','wait_and_throw()','ext_oneapi_empty','memcpy('):self.assertEqual(source.count(name),out.count(name))
  self.assertIn('host_stage.returned(host_stage_ok)',out);self.assertIn('host_stage.submitted(le==0)',out)
  self.assertLess(out.index('host_stage.submitting();'),out.index('const dpct::err0 le =',out.index('host_stage.submitting();')))
  hdr=Path(p.__file__).parent/'original-mtp53/phase_timing_v2.hpp';s=hdr.read_text();self.assertIn('capacity=32768',s);self.assertIn('rows.reserve(capacity)',s);self.assertNotIn('sycl::',s);self.assertIn('spans_additive=0',s)
if __name__=='__main__':unittest.main()
