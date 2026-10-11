import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import build_pti_paused_regular_lists_v2 as b
class Tests(unittest.TestCase):
 def test_popen_failure_still_recovers_attempted_name(self):
  with tempfile.TemporaryDirectory()as t:
   out=Path(t)/'new';calls=[]
   cmd=['docker','create','--name','owned','image','-c','true']
   with patch.object(b,'OUTPUT',out),patch.object(b,'materialize',return_value={}),patch.object(b.p,'source_binding',return_value={}),patch.object(b.recipe,'recipe',return_value=cmd),patch.object(b,'validate'),patch.object(b.signal,'signal',return_value=None),patch.object(b.subprocess,'Popen',side_effect=OSError('injected launch ambiguity')),patch.object(b,'recover_created',side_effect=lambda c:(calls.append(c)or(None,{'return_code':1,'stderr':'No such object'}))):
    with self.assertRaises(ValueError):b.main()
   self.assertEqual(calls,[cmd]);row=json.loads((out/'receipt.json').read_text());self.assertFalse(row['passed']);self.assertTrue(row['original_create_attempted']);self.assertFalse(row['original_creator_handle_returned']);self.assertIn('injected launch ambiguity',row['error'])
 def test_sighup_normal_and_cleanup_handlers(self):
  text=Path(b.__file__).read_text();self.assertEqual(text.count('(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)'),2)
if __name__=='__main__':unittest.main()
