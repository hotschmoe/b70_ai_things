"""Fresh oracle generation controls with external metadata boundaries mocked."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import admit_source_upload37_v1 as a
class Controls(unittest.TestCase):
 def setUp(self):self.g={'plan_sha256':a.c.COMBINED_PLAN_SHA,'engine_receipt_sha256':'CPU_NEW_ENGINE','engine_receipt':'/CPU_SOURCE37/receipt.json'}
 def test_new_recipe_metadata_only_positive(self):
  with patch.object(a.c,'combined_generation_gate',return_value=self.g):self.assertFalse(a.admission('/CPU_SOURCE37')['actual_upload_qualified'])
 def test_old_engine_refused_before_any_oracle_read(self):
  with patch.object(a.c,'combined_generation_gate',side_effect=ValueError('old35')),patch.object(a.c,'read') as read:
   with self.assertRaises(ValueError):a.admission('/CPU_SOURCE35','/CPU_OLD_ORACLE/receipt.json')
   read.assert_not_called()
 def test_old_oracle_recipe_refused_even_passed_flag(self):
  oracle=Path('/CPU_OLD_ORACLE/receipt.json');recipe=a.c.read(a.PLAN);r={'passed':True,'build_rc':0,'libraries_unchanged':True,'image':a.c.BASE_IMAGE,'engine_receipt_sha256':'CPU_OLD_ENGINE','engine_receipt':'/CPU_SOURCE35/receipt.json','plan_sha256':'CPU_OLD_PLAN'}
  with patch.object(a.c,'combined_generation_gate',return_value=self.g),patch.object(a.c,'read',side_effect=lambda p:recipe if Path(p)==a.PLAN else r):
   with self.assertRaises(ValueError):a.admission('/CPU_SOURCE37',oracle)
 def test_h36_ON_recipe_refused(self):
  recipe=a.c.read(a.PLAN);recipe['runtime_environment']['STRATA_CRITICAL_PATH_TRACE']='1'
  with patch.object(a.c,'combined_generation_gate',return_value=self.g),patch.object(a.c,'read',return_value=recipe):
   with self.assertRaises(ValueError):a.admission('/CPU_SOURCE37')
if __name__=='__main__':unittest.main()
