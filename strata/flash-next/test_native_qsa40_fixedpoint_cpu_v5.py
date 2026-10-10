"""Exact file-only nested metadata chain and bounded complete traversal."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_qsa40_byte_epoch_v5 as e
class Controls(unittest.TestCase):
 def tree(self,root):
  prepared=root/'prepared';prepared.mkdir();external=root/'external';external.mkdir();outside=root/'outside.bin';outside.write_bytes(b'CPU_RAW');child=external/'child.json';child.write_text(json.dumps({'actual_raw_path':str(outside)}));(prepared/'report.json').write_text(json.dumps({'referenced_report_path':str(child)}));plan={key:str(prepared)for key in('prepared','engine_root','pack','own_producer_root')};return plan,prepared,external,child,outside
 def test_exact_peer_external_file_only_chain_reaches_fixedpoint(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,prepared,external,child,outside=self.tree(root)
   with patch.object(e,'PREFIXES',(root,)),patch.object(e,'current_row',side_effect=AssertionError('Metadata only')):
    roots,files=e.roots_and_files(plan,[])
   self.assertEqual(roots,[prepared]);self.assertIn(child,files);self.assertIn(outside,files);self.assertNotIn(external,roots)
 def test_referenced_child_oversize_refused_before_parse(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,prepared,external,child,outside=self.tree(root);(prepared/'report.json').write_text('{"path":"'+str(child)+'"}');child.write_text(' '*200)
   actual=e.read_unique;parsed=[]
   def read(path):parsed.append(Path(path));return actual(path)
   with patch.object(e,'PREFIXES',(root,)),patch.object(e,'MAX_JSON_BYTES',180),patch.object(e,'read_unique',side_effect=read):self.assertRaises(ValueError,e.roots_and_files,plan,[])
   self.assertNotIn(child,parsed)
 def test_file_cycle_reaches_fixedpoint_without_skipping_raw_leaf(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,prepared,external,child,outside=self.tree(root);child.write_text(json.dumps({'report_path':str(prepared/'report.json'),'raw_path':str(outside)}))
   with patch.object(e,'PREFIXES',(root,)):roots,files=e.roots_and_files(plan,[])
   self.assertIn(outside,files);self.assertEqual(len(set(files)),len(files))
 def test_authenticated_plan_only_file_and_nested_JSON_leaf_are_covered(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);plan,prepared,external,child,outside=self.tree(root);(prepared/'report.json').write_text('{}');plan['plan_only_evidence_path']=str(child);plan['shell']='/bin/bash'
   with patch.object(e,'PREFIXES',(root,)):roots,files=e.roots_and_files(plan,[])
   self.assertIn(child,files);self.assertIn(outside,files);self.assertNotIn(Path('/bin/bash'),files)
if __name__=='__main__':unittest.main()
