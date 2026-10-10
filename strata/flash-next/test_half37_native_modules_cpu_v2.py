"""Independent ordered source/owner/count/binary controls, no runtime/tool calls."""
import copy,json,tempfile,unittest
from pathlib import Path
import collect_half37_native_modules_v2 as c
class Controls(unittest.TestCase):
 def fixture(self,root):
  rows=[{'event':'proc_table','name':'zeGetModuleProcAddrTable','version':65536}];(root/'module-1-input.bin').write_bytes(b'original synthetic input');(root/'module-1-native.bin').write_bytes(b'\x7fELF'+b'CPU_SYNTHETIC');rows += [{'event':'module','id':1,'context':'0x1','device':'0x2','device_properties_rc':0,'input':'module-1-input.bin','native':'module-1-native.bin'},{'event':'kernel','id':2,'module':1,'name':'CPU_SYNTHETIC'},{'event':'append','kernel':2,'list':3},{'event':'submitted','kernel':2,'module':1,'name':'CPU_SYNTHETIC','list':3,'via':'queue'}];rows.append({'event':'terminal','failed':False,'events':5,'binary_bytes':sum(p.stat().st_size for p in root.iterdir())})
  for n,row in enumerate(rows,1):row.update(sequence=n,pid=123)
  self.write(root,rows);return rows
 def write(self,root,rows):(root/'events.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
 def test_structural_actual_bytes_and_dispatch_are_not_GPU_authority(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.fixture(root);p=c.structural_binding(root,synthetic=True);self.assertEqual(p['submitted'],{'CPU_SYNTHETIC':1});self.assertFalse(p['actual_GPU_execution_proven_by_this_structural_reader'])
 def test_missing_kernel_module_or_foreign_submit_rejects(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);original=self.fixture(root)
   for at,key,value in [(2,'module',99),(4,'module',99),(4,'name','FOREIGN'),(4,'kernel',100),(5,'failed',True),(5,'events',0),(5,'binary_bytes',0),(0,'pid',True),(0,'sequence',True)]:
    rows=copy.deepcopy(original);rows[at][key]=value;self.write(root,rows);self.assertRaises(ValueError,c.structural_binding,root,synthetic=True)
 def test_exclusive_roster_and_nonalias_binary_required(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.fixture(root);(root/'foreign').write_bytes(b'');self.assertRaises(ValueError,c.structural_binding,root,synthetic=True);(root/'foreign').unlink();p=root/'module-1-native.bin';raw=p.read_bytes();p.unlink();other=root.parent/(root.name+'-code');other.write_bytes(raw);p.symlink_to(other)
   try:self.assertRaises(ValueError,c.structural_binding,root,synthetic=True)
   finally:other.unlink()
 def test_explicit_capture_error_and_missing_terminal_reject(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);self.write(root,rows[:-1]);self.assertRaises(ValueError,c.structural_binding,root,synthetic=True);rows[-1]['event']='error';self.write(root,rows);self.assertRaises(ValueError,c.structural_binding,root,synthetic=True)
if __name__=='__main__':unittest.main()

class SpecificHandle(Controls):
 def test_specific_symbol_lookup_binds_actual_library_and_typed_name(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);rows=self.fixture(root);rows.insert(0,{'event':'symbol_lookup','name':'zeGetModuleProcAddrTable','library':'/out/CPU_SYNTHETIC_fake_driver.so'})
   for n,row in enumerate(rows,1):row.update(sequence=n,pid=123)
   rows[-1]['events']+=1;self.write(root,rows);self.assertEqual(c.structural_binding(root,synthetic=True)['PID'],123)
   for key,value in [('name','FOREIGN'),('library','relative'),('name',True)]:
    bad=copy.deepcopy(rows);bad[0][key]=value;self.write(root,bad);self.assertRaises(ValueError,c.structural_binding,root,synthetic=True)
