import ast,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pti_mapped_injection_census_v2 as q
class Controls(unittest.TestCase):
 def fixture(self,root):
  proc=root/'proc';proc.mkdir();(proc/'map_files').mkdir();libroot=root/'install';(libroot/'lib').mkdir(parents=True);lib=libroot/'lib/libunitrace_tool.so';lib.write_bytes(b'\x7fELF'+bytes(4092));s=lib.stat();statline='1 (tiny native) S '+' '.join(['0']*18+['99']+['0']*4);(proc/'stat').write_text(statline);line='1000-2000 r-xp 00000000 00:32 %d %s\n'%(s.st_ino,lib);(proc/'maps').write_text(line);(proc/'map_files/1000-2000').symlink_to(lib);(proc/'mountinfo').write_text('711 1 0:50 /b70/install %s ro - btrfs /dev/sdc1 rw,subvolid=5\n'%libroot);return proc,lib,libroot
 def run_case(self,kind):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);proc,lib,mp=self.fixture(root)
   if kind=='inode':(proc/'maps').write_text((proc/'maps').read_text().replace(str(lib.stat().st_ino),'999'))
   if kind=='mount':(proc/'mountinfo').write_text((proc/'mountinfo').read_text().replace('0:50','0:51'))
   if kind=='fs':(proc/'mountinfo').write_text((proc/'mountinfo').read_text().replace('btrfs','fuse.mergerfs'))
   if kind=='link':(proc/'map_files/1000-2000').unlink();(proc/'map_files/1000-2000').symlink_to(root/'foreign')
   sha=hashlib.sha256(lib.read_bytes()).hexdigest()
   if kind=='sha':sha='b'*64
   def action():return q.capture(root/'new',sha,'after_READY','/b70/install','/dev/sdc1',proc,lib,str(mp))
   if kind=='valid':
    r=action();self.assertEqual(r['schema'],2);self.assertEqual(r['backing_file_stat'],r['opened_fd_stat_before']);self.assertEqual(r['btrfs_device_domain_association']['mount_device'],[0,50]);self.assertFalse(r['process_memory_bytes_read']);self.assertEqual((root/'new/mapped-injection.elf').read_bytes(),lib.read_bytes())
   else:
    with self.assertRaises(ValueError):action()
 def test_actual_tiny_fd_hash_mapping_producer(self):self.run_case('valid')
 def test_wrong_inode(self):self.run_case('inode')
 def test_wrong_mountdev(self):self.run_case('mount')
 def test_wrong_fs(self):self.run_case('fs')
 def test_foreign_maplink(self):self.run_case('link')
 def test_wrong_elf_hash(self):self.run_case('sha')
 def test_standalone_copy_of_original_and_association_predicates(self):
  import pti_mapped_injection_census_v1 as old
  import pti_btrfs_backing_identity_proposal_v1 as proposal
  current={n.name:ast.dump(n)for n in ast.parse(Path(q.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
  for module,names in((old,('require','signature','process','select')),(proposal,('unescape','covering_mount','associate'))):
   source={n.name:ast.dump(n)for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
   for name in names:self.assertEqual(current[name],source[name])
 def test_changed_mapping_after_opened_fd_hash_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);proc,lib,mp=self.fixture(root);real=q.process;calls=[]
   def process(text):
    calls.append(1)
    if len(calls)==2:(proc/'maps').write_text((proc/'maps').read_text().replace('1000-2000','3000-4000'))
    return real(text)
   with patch.object(q,'process',side_effect=process):
    with self.assertRaises(ValueError):q.capture(root/'new',hashlib.sha256(lib.read_bytes()).hexdigest(),'after_READY','/b70/install','/dev/sdc1',proc,lib,str(mp))
 def test_backing_stat_changes_after_hash_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);proc,lib,mp=self.fixture(root);real=q.signature;calls=[]
   def signature(path):
    value=real(path);calls.append(1)
    if len(calls)>=2:value=value.copy();value[4]+=1
    return value
   with patch.object(q,'signature',side_effect=signature):
    with self.assertRaises(ValueError):q.capture(root/'new',hashlib.sha256(lib.read_bytes()).hexdigest(),'after_READY','/b70/install','/dev/sdc1',proc,lib,str(mp))
 def test_library_alias_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);proc,lib,mp=self.fixture(root);alias=lib.parent/'alias';alias.symlink_to(lib)
   with self.assertRaises(ValueError):q.capture(root/'new','a'*64,'after_READY','/b70/install','/dev/sdc1',proc,alias,str(mp))
if __name__=='__main__':unittest.main()
