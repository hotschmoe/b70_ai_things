import copy,unittest
from unittest.mock import patch
import native52_startup_option_preflight_v1 as a
class Controls(unittest.TestCase):
 def recipe(self):return {'native_argv':['/pti/bin/unitrace','--chrome-kernel-logging','/build/strata','--serve','--batch','0','--spec','2','--pipeline-windows','0','--suffix-draft','0','--lookup-chain','0','--layer-split','24','--split-device','1'],'native_env':{'STRATA_CRITICAL_PATH_TRACE':'0','STRATA_VERIFY_MIRROR_STAGE_T1':'1'}}
 def test_pair_H36_OFF_and_real_failure_ON(self):
  with patch.object(a,'source_binding',return_value={}):
   r=self.recipe();self.assertTrue(a.check(r,'/actual')['passed']);r['native_env']['STRATA_CRITICAL_PATH_TRACE']='1'
   with self.assertRaisesRegex(ValueError,'BEFORE model'):a.check(r,'/actual')
 def test_serial_enabled_declarations_and_eager_zero(self):
  with patch.object(a,'source_binding',return_value={}):
   r=self.recipe();r['native_argv']=r['native_argv'][:-4];r['native_env'].update(STRATA_CRITICAL_PATH_TRACE='1',STRATA_CRITICAL_PATH_TRACE_DIR='/results/host-traces',STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256='a'*64);self.assertTrue(a.check(r,'/actual')['H36_enabled']);r['native_env']['STRATA_VERIFY_EAGER']='0'
   with self.assertRaises(ValueError):a.check(r,'/actual')
 def test_current_small_source_anchor(self):
  self.assertEqual(len(a.source_binding('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source')['files']),2)
if __name__=='__main__':unittest.main()
