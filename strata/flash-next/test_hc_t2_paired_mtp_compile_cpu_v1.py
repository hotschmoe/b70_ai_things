import unittest
import prepare_hc_t2_paired_mtp_compile_v1 as m
class Recipe(unittest.TestCase):
 def test_modular_link(self):
  src='/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp'
  r={'compile_commands':[{'argv':['icpx','-fsycl',src,'-Wl,--start-group','/sdk/build/libstrata_engine.a','/sdk/build/libstrata_kernels.a','/sdk/build/libstrata_core.a','-Wl,--end-group','-o','/out/old']}],'production_link_flags':'-fsycl -qmkl=sequential'}
  cmds,link,filters=m.modularize(r)
  self.assertEqual(len(cmds),4);self.assertNotIn('/sdk/build/libstrata_engine.a',link);self.assertNotIn('/sdk/build/libstrata_kernels.a',link)
  self.assertLess(link.index('/out/verify.hc-t2.o'),link.index('-Wl,--start-group'))
  self.assertEqual(filters[0]['remove_members'],['verify.cpp.o']);self.assertEqual(filters[1]['after_counts'],[0,0])
  self.assertEqual(r['compile_commands'][0]['argv'][-1],'/out/old')
 def test_wrong_parent_refused(self):
  with self.assertRaises(ValueError):m.modularize({'compile_commands':[{'argv':['icpx','other']} ]})
