import hashlib,json,os,tempfile,unittest
from pathlib import Path
import pti_mapped_injection_census_v1 as c
import pti_census_owned_exec_v1 as e
class Controls(unittest.TestCase):
 def fixture(self,root):
  proc=root/'proc';proc.mkdir();lib=root/'libunitrace_tool.so';lib.write_bytes(b'\x7fELFsynthetic-owned-backing');st=lib.stat();tail=['S']+['0']*18+['123']+['0']*5;(proc/'stat').write_text('1 (strata) '+' '.join(tail));maps=f'1000-2000 r-xp 00000000 {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x} {st.st_ino} {lib}\n';(proc/'maps').write_text(maps);return proc,lib,hashlib.sha256(lib.read_bytes()).hexdigest()
 def test_actual_tiny_backing_export_identity_and_no_memory_claim(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);proc,lib,sha=self.fixture(root);proof=c.capture(root/'out',sha,'after_READY',proc,lib);self.assertTrue(proof['passed']);self.assertEqual((root/'out/mapped-injection.elf').read_bytes(),lib.read_bytes());self.assertFalse(proof['process_memory_bytes_read']);self.assertFalse(proof['versioned_symbol_origin_queried'])
 def test_wrong_mapping_inode_deleted_and_digest_refused(self):
  for kind in('inode','deleted','sha'):
   with tempfile.TemporaryDirectory()as t:
    root=Path(t);proc,lib,sha=self.fixture(root)
    if kind=='inode':(proc/'maps').write_text((proc/'maps').read_text().replace(' '+str(lib.stat().st_ino)+' ',' 999 '))
    elif kind=='deleted':(proc/'maps').write_text((proc/'maps').read_text().rstrip()+' (deleted)\n')
    else:sha='0'*64
    with self.subTest(kind=kind),self.assertRaises(ValueError):c.capture(root/'out',sha,'after_READY',proc,lib)
 def test_exact_owned_exec_phase_recipe_no_model_request(self):
  command=e.command('owned-id','after_last_DONE_before_QUIT','a'*64);self.assertEqual(command[:3],['docker','exec','owned-id']);self.assertIn('/pti-census.py',command);self.assertNotIn('GEN',command)
  with self.assertRaises(ValueError):e.command('owned-id','during_kernel','a'*64)
if __name__=='__main__':unittest.main()
