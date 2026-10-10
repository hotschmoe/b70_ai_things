"""Actual small code metadata and translator option controls only."""
import tempfile,unittest
from pathlib import Path
import half37_spirv_text_recipe_v3 as p
class Controls(unittest.TestCase):
 def test_actual_six_saved_bitcode_and_symbols_bound_to_successful_capture(self):
  b=p.original_modules();self.assertEqual([row['index']for row in b['modules']],[21,24,25,33,36,38]);self.assertTrue(all(row['bytes']>0 for row in b['modules']));self.assertFalse(b['original_runtime_JIT_ISA_observed'])
 def test_actual_translator_options_preserved_separately_from_backend_JIT_options(self):
  options=p.translator_options(p.PROBES/'save-temps-driver-phases.log');self.assertIn('-spirv-max-version=1.5',options);self.assertTrue(any(x.startswith('-spirv-ext=')for x in options));self.assertNotIn('-cl-fp32-correctly-rounded-divide-sqrt',options)
 def test_missing_duplicate_or_invented_translator_phase_refused(self):
  line='"foreach" "--out-ext=spv" "'+p.inventory.INSTALLED+'llvm-spirv" "-o" "OUTPUT" "-spirv-max-version=1.5" "-spirv-ext=-all" "INPUT"\n'
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'log'
   for text in ('none\n',line+line,line.replace('-spirv-max-version=1.5','-GUESS')):
    path.write_text(text);self.assertRaises(ValueError,p.translator_options,path)
if __name__=='__main__':unittest.main()
