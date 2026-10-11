"""Every referenced/auth JSON leaf is checked before any target read."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_qsa40_byte_epoch_v7 as e
class Controls(unittest.TestCase):
 def test_glob_symlink_JSON_never_enters_parser(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir();target=root/'target.json';target.write_text('{}');(data/'linked.json').symlink_to(target);plan={k:str(data)for k in('prepared','engine_root','pack','own_producer_root')}
   with patch.object(e,'PREFIXES',(root,)),patch.object(e,'read_unique')as parser:self.assertRaises(ValueError,e.roots_and_files,plan,[]);parser.assert_not_called()
 def test_auth_prepared_leaf_symlink_rejected_before_target_read(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir();target=root/'target.json';target.write_text('{}');(data/'prepared.json').symlink_to(target);plan={k:str(data)for k in('prepared','engine_root','pack','own_producer_root')};plan['prepared_sha256']='0'*64
   with patch.object(Path,'read_bytes',side_effect=AssertionError('No linked bytes may be read')):self.assertRaises(ValueError,e.roots_and_files,plan,[])
 def test_cheap_preparation_own_report_alias_refused_before_semantics(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);prepared=root/'prepared';own=root/'own';prepared.mkdir();own.mkdir();(prepared/'prepared.json').write_text('{}');target=root/'target.json';target.write_text('{}');(own/'report.json').symlink_to(target)
   with patch.object(e,'read_unique')as parser:self.assertRaises(ValueError,e.metadata_preflight,prepared,own);parser.assert_not_called()
if __name__=='__main__':unittest.main()
