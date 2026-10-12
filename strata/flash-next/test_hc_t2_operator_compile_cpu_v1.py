import unittest
from pathlib import Path
import prepare_hc_t2_operator_compile_v1 as p
class LeafRecipe(unittest.TestCase):
 def test_fresh_objects_filter_and_scope(self):
  s=Path(p.__file__).read_text()
  for name in ('hc_native_projection.cpp','hc_native_composition.cpp','verify.cpp','fixture.cpp'):self.assertIn(name,s)
  self.assertIn("'verify_object_compiled_but_not_linked_in_leaf':True",s)
  self.assertIn("'remove_members':['hc_native_projection.cpp.o','hc_native_composition.cpp.o']",s)
  self.assertIn("'/out/hc_projection.t2.o','/out/hc_composition.t2.o','-Wl,--start-group'",s)
  self.assertIn("'NoDri':True",s);self.assertIn("'model_mounted_or_read':False",s)
