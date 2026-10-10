"""Metadata/source recipe checks only; no compiler/IR/native execution."""
import unittest
import half37_compiler_lowering_recipe_v1 as p
class Controls(unittest.TestCase):
 def test_exact_original_math_compile_flags_with_only_artifact_emission_changes(self):
  a=p.flags()['original_compile_argv'];b=p.llvm_argv();self.assertIn('-fp-model=precise',b);self.assertIn('-O3',b);self.assertIn('-fsycl-default-sub-group-size=32',b);self.assertIn('/leaf/owned_indexer_half_discrimination_gpu_v2.cpp',b);self.assertEqual(b[-5:],['-fsycl-device-only','-S','-emit-llvm','-o','/out/half37-device.ll']);self.assertNotIn('/sdk/build/libstrata_kernels.a',b);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',a);self.assertNotIn('-cl-fp32-correctly-rounded-divide-sqrt',b)
 def test_missing_tool_inventory_is_truthful_not_invented(self):
  s=p.tool_inventory_shell();self.assertIn('TOOL_UNAVAILABLE_ocloc',s);self.assertIn('SETVARS_COMPLETED',s);self.assertIn('command -v llvm-spirv',s)
 def test_downstream_separates_fresh_compilation_from_original_ELF_and_unobserved_ISA(self):
  r=p.downstream('/CPU/llvm-as','/CPU/llvm-spirv','/CPU/spirv-dis','/CPU/llvm-readelf','/original/helper');self.assertTrue(r['original_ELF_embedded_SPIRV_extraction_recipe_pending_actual_section_inventory']);self.assertFalse(r['actual_runtime_JIT_ISA_saved']);self.assertTrue(r['offline_ISA_cannot_be_called_original_runtime_ISA']);self.assertEqual(r['original_executed_ELF_section_inventory'][-1],'/original/helper')
 def test_unobserved_or_relative_tool_paths_refused(self):
  self.assertRaises(ValueError,p.downstream,'llvm-as','/CPU/tool','/CPU/tool','/CPU/tool','/original/helper')
if __name__=='__main__':unittest.main()
