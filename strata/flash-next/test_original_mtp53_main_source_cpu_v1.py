import unittest
from pathlib import Path
import prepare_original_mtp53_main_compile_v1 as m
class MainSource(unittest.TestCase):
 def test_exact_native_hook_and_failure_retirement(self):
  source='start\n    strata::core::MtpDrafter mtp;\n'+('        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);\n'*2)+'end\n'
  result=m.transform(source)
  self.assertEqual(result.count('original53->exercise(ver,win_pool_fn,win_pool_user)'),1)
  self.assertIn('|| drive.d.failed',result);self.assertIn('std::_Exit(86)',result)
  self.assertIn('o.spec!=2',result);self.assertIn('n_stages!=1',result)
  self.assertIn('o.pipeline_windows!=0',result)
  self.assertEqual(result.count('strata::core::MtpDrafter mtp;'),1)
  self.assertEqual(result.count('ver.set_pcie_mode(o.pcie_mode'),2)
  with self.assertRaises(ValueError):m.transform(source.replace('MtpDrafter','Unknown'))
 def test_actual_main_control_uses_authoritative_picks(self):
  s=(m.HERE/'original-mtp53/loaded_main_adapter_v1.hpp').read_text()
  self.assertIn('main.run(1,&input,0',s);self.assertIn('main.commit(1,error)',s)
  self.assertIn('catchup(next,main.final_R_all())',s)
  self.assertIn('draft->verify_and_catchup(main',s)
  self.assertIn('actual[0]!=expected',s);self.assertIn('state->committed_cells()!=position',s)
  self.assertIn('trial==0?(expected+1)%248320:trial==1?expected:proposed',s)
  self.assertIn('failed=true;',s);self.assertIn('full64_equivalence=0',s)
