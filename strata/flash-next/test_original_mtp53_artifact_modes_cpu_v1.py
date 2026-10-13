import unittest
from prepare_original_mtp53_artifact_identity_v1 import modes
class Modes(unittest.TestCase):
 def test_real_publication_restart_env_is_same(self):
  r={'native_argv':['/original-serve/original-mtp53-serve','--serve','--spec','2'],'native_env':{'STRATA_ORIGINAL_MTP53':'1','STRATA_ARTIFACT_IDENTITY_SHA256':'a'*64}}
  a=modes(r);self.assertNotIn('STRATA_ARTIFACT_IDENTITY_SHA256',a['1']['env']);self.assertEqual(a['1'],modes(r)['1']);self.assertEqual(a['0']['args'],a['1']['args']);self.assertEqual(a['1']['env']['STRATA_ORIGINAL_MTP53_SERVING'],'1')
