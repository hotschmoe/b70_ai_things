"""Offline target/input/recipe controls; no compiler, GPU or model calls."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import half37_original_offline_ocloc_recipe_v1 as r
class Controls(unittest.TestCase):
 def test_only_actual_observed_original_image_indices_and_current_hex(self):
  for index in r.original.MODULES:
   argv=r.compile_argv(index);self.assertEqual(argv[argv.index('-device')+1],'0xe223');self.assertIn('-spirv_input',argv);self.assertEqual(argv[argv.index('-options')+1],r.OPTION);self.assertNotIn('pvc',argv);self.assertNotIn('bmg',argv)
  for index in (True,0,100):self.assertRaises(ValueError,r.compile_argv,index)
 def test_all_six_bounded_owned_compile_recipes_and_actual_loader_output(self):
  with patch.object(r,'evidence_binding',return_value={'observed':True}),patch.object(r,'input_binding',return_value={'current_original_six_symbol_byte_join':True}),patch.object(r,'current_host_target',return_value={'target_hex':r.TARGET}):result=r.compile_recipes('/tmp/owned-future-output',123)
  self.assertEqual(set(result['commands']),{str(i)for i in r.original.MODULES});self.assertFalse(result['original_JIT_ISA_or_lowering_cause_qualified']);self.assertFalse(result['kernel_or_model_execution_requested']);self.assertFalse(result['all_original_runtime_backend_options_identical'])
  for index,command in result['commands'].items():self.assertIn('LD_DEBUG=libs,files',command);self.assertIn('LD_DEBUG_OUTPUT=/out/module-'+index+'-loader',command);self.assertIn('--entrypoint',command);self.assertIn('2g',command);self.assertNotIn('/pack',str(command));self.assertNotIn('/model',str(command));self.assertIn('/images/module-'+index+'.spv',command[-1])
 def test_root_bool_refused_before_any_provenance_read(self):
  with patch.object(r,'evidence_binding')as read:self.assertRaises(ValueError,r.compile_recipes,'/tmp/not-created',True);read.assert_not_called()
 def test_disassembly_requires_complete_observed_output_roster(self):
  with patch.object(r,'evidence_binding'),patch.object(r,'input_binding'),patch.object(r,'current_host_target',return_value={}):self.assertRaises(ValueError,r.disasm_recipes,{'33':{}},'/tmp/not-created',123)
 def test_disassembly_never_guesses_output_binary_from_spirv_basename(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);manifest={};inputs={}
   for index in r.original.MODULES:
    file=root/('actually-observed-'+str(index)+'.bin');file.write_bytes(b'\x7fELF'+bytes([index]));inputs['module-'+str(index)+'.spv']='a'*64;manifest[str(index)]={'path':str(file),'sha256':r.original.digest(file.read_bytes()),'bytes':file.stat().st_size,'input_sha256':'a'*64,'target_hex':r.TARGET}
   with patch.object(r,'evidence_binding'),patch.object(r,'input_binding'),patch.object(r,'current_host_target',return_value={}),patch.object(r,'sha',side_effect=lambda p:inputs[Path(p).name]):
    result=r.disasm_recipes(manifest,root/'new',123);self.assertEqual(len(result['commands']),6)
    for index,command in result['commands'].items():self.assertIn('/input/actually-observed-'+index+'.bin',command[-1]);self.assertIn('-dump /out/module-'+index,command[-1])
    manifest['33']['input_sha256']='0'*64;self.assertRaises(ValueError,r.disasm_recipes,manifest,root/'new',123)
if __name__=='__main__':unittest.main()
