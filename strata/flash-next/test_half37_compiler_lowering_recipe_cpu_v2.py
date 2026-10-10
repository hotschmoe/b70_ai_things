"""Actual inventory metadata and synthetic phase controls; no compilation."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import half37_compiler_lowering_recipe_v2 as p
class Controls(unittest.TestCase):
 def test_actual_absolute_observed_tools_and_no_absence_inference_elsewhere(self):
  r=p.inventory_binding();self.assertEqual(len(r['tools']),7);self.assertTrue(r['other_tool_locations_not_inventoried']);self.assertEqual(r['tools'][p.INSTALLED+'clang++']['resolved'],p.INSTALLED+'clang')
 def test_probes_keep_all_original_compile_and_backend_flags(self):
  rows=p.probe_argv();old=p.q.leaf_argv();self.assertEqual(rows['actual-driver-phases'],[old[0],'-###',*old[1:]]);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',rows['save-temps-driver-phases']);self.assertIn(p.INSTALLED+'llvm-spirv',rows['llvm-spirv-help']);self.assertNotIn('-emit-llvm',rows['save-temps-driver-phases'])
 def fixture(self,tmp):
  path=Path(tmp)/'phase.log';path.write_text('"'+p.INSTALLED+'clang" "-cc1" "-fsycl-is-device" "-O3" "/leaf/owned_indexer_half_discrimination_gpu_v2.cpp"\nllvm-spirv sycl-post-link -cl-fp32-correctly-rounded-divide-sqrt\n');row={'return_code':0,'passed':True,'eof':True,'reader_retired':True,'error':None,'command_error':None,'phase_cleanup_error':None,'sha256':p.q.sha(path)};return path,row
 def test_supported_actual_phase_authorizes_only_fresh_save_temps_recompile(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);path,row=self.fixture(tmp);path.rename(root/'save-temps-driver-phases.log');row['command']=['CPU_EXACT'];p.q.write(root/'save-temps-driver-phases.receipt.json',row);p.q.write(root/'save-temps-driver-phases.inspection.json',{'State':{},'CPU':'fixture'})
   original=p.q.leaf_argv()
   with patch.object(p.q,'leaf_argv',return_value=original),patch.object(p,'recipes',return_value={'save-temps-driver-phases':['CPU_EXACT']}),patch.object(p.q,'command_binding'),patch.object(p.q,'terminal_binding'),patch.object(p.q,'observed_container'),patch.object(p.q,'modules',return_value=(type('CPU',(),{'IMAGE':'CPU'})(),)),patch.object(p.q,'read',return_value={}):
    # Exact metadata phase structure is tested independently; no container exists.
    with patch.object(p,'recipes',return_value={'save-temps-driver-phases':['CPU','--name','synthetic']}):
     row['command']=['CPU','--name','synthetic'];p.q.write(root/'save-temps-driver-phases.receipt.json',row);a=p.save_temps_argv(root,17)
   self.assertIn('-save-temps=obj',a);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',a);self.assertEqual(a[-1],'/out/half37-save-temps');self.assertTrue(p.device_frontend(root/'save-temps-driver-phases.log')['actual_link_backend_options_not_equated_with_frontend'])
 def test_failed_saved_receipt_or_foreign_full_recipe_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);path,row=self.fixture(tmp);path.rename(root/'save-temps-driver-phases.log');row['command']=['CPU_FOREIGN'];p.q.write(root/'save-temps-driver-phases.receipt.json',row)
   with patch.object(p,'recipes',return_value={'save-temps-driver-phases':['CPU_EXPECTED']}),patch.object(p.q,'command_binding'):
    self.assertRaises(ValueError,p.save_temps_argv,root,17)
 def test_missing_duplicate_or_foreign_device_phase_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   path,row=self.fixture(tmp);line=path.read_text().splitlines()[0]
   for text in ('none\n',line+'\n'+line+'\n',line.replace(p.INSTALLED+'clang','/foreign/clang')+'\n'):
    path.write_text(text);self.assertRaises(ValueError,p.device_frontend,path)
 def test_isa_and_original_embedded_code_are_not_inferred_from_fresh_compilation(self):
  r=p.downstream_probes('/original/helper');self.assertTrue(r['fresh_save_temps_recompile_is_not_original_executed_ELF']);self.assertTrue(r['original_ELF_extract_argv_pending_observed_help']);self.assertFalse(r['actual_original_runtime_JIT_ISA_saved'])
if __name__=='__main__':unittest.main()
