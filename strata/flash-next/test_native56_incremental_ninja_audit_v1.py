"""Tiny synthetic dependency controls; no compiler, native process or model access."""
import copy, unittest
from native56_incremental_ninja_audit_v1 import audit

class Controls(unittest.TestCase):
 def inputs(self):
  obj='CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o'
  proof=dict(changed_object=obj,object_sha256={obj:'new','other.o':'old'},source_tree={'sycl/src/kernels/hc_native_composition.cpp':'body','sycl/include/x.hpp':'header'},ggml_tree={},build_dependency_sha256={})
  deps=obj+': #deps 2, deps mtime 0 (VALID)\n    /src/sycl/src/kernels/hc_native_composition.cpp\n    /src/sycl/include/x.hpp\n\nother.o: #deps 1, deps mtime 0 (VALID)\n    /src/sycl/include/x.hpp\n'
  commands=['icpx -o '+obj+' -c /src/sycl/src/kernels/hc_native_composition.cpp','llvm-ar qc libstrata_kernels.a '+obj]+['icpx -o elf%d other.o'%i for i in range(8)]
  dry='\n'.join('[%d/10] %s'%(i+1,x) for i,x in enumerate(commands))
  return proof,deps,dry
 def test_only_changed_body_is_rebuilt(self):
  p,d,r=self.inputs();self.assertTrue(audit(p,d,r,lambda _: 'header')['passed'])
 def test_header_change_and_extra_compile_rejected(self):
  p,d,r=self.inputs()
  with self.assertRaisesRegex(ValueError,'header dependency changed'):audit(p,d,r,lambda _: 'foreign')
  with self.assertRaisesRegex(ValueError,'exactly the changed composition'):audit(p,d,r+'\n[11/11] icpx -o other.o -c /src/other.cpp',lambda _: 'header')
if __name__=='__main__':unittest.main()
