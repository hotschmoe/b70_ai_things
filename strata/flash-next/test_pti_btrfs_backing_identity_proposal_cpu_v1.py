import copy,unittest
import pti_btrfs_backing_identity_proposal_v1 as q
class Controls(unittest.TestCase):
 def fixture(self):
  text='1 0 0:66 / / rw - overlay overlay rw\n711 1 0:50 /b70/install /pti ro,noatime - btrfs /dev/sdc1 rw,subvolid=5,subvol=/\n';m=q.covering_mount(text,'/pti/lib/libunitrace_tool.so');s=[45,17686394,3818408,2,3];r=[{'start':4096,'end':3825664,'offset':0,'inode':s[1],'device_major':0,'device_minor':50,'original_container_path':'/pti/lib/libunitrace_tool.so','permissions':'r-xp'}];return [r,s,s,s,s,m,m,'/pti/lib/libunitrace_tool.so','/b70/install','/dev/sdc1','a'*64,'a'*64,['/pti/lib/libunitrace_tool.so']]
 def test_affirmative_two_domain_identity(self):
  r=q.associate(*self.fixture());self.assertFalse(r['arbitrary_device_mismatch_accepted']);self.assertFalse(r['actual_producer_or_runtime_qualified'])
 def test_wrong_inode_refused(self):
  a=self.fixture();a[0][0]['inode']+=1
  with self.assertRaises(ValueError):q.associate(*a)
 def test_mapdev_must_equal_mount_not_arbitrary(self):
  a=self.fixture();a[0][0]['device_minor']=51
  with self.assertRaises(ValueError):q.associate(*a)
 def test_foreign_filesystem_or_bindroot_refused(self):
  for key,value in [('filesystem','fuse.mergerfs'),('root','/other/install'),('source','/dev/other')]:
   a=self.fixture();a[5]=copy.deepcopy(a[5]);a[6]=a[5];a[5][key]=value
   with self.assertRaises(ValueError):q.associate(*a)
 def test_deleted_maplink_refused(self):
  a=self.fixture();a[-1]=[a[7]+' (deleted)']
  with self.assertRaises(ValueError):q.associate(*a)
 def test_openedFD_must_equal_path(self):
  a=self.fixture();a[3]=a[3].copy();a[3][0]=51
  with self.assertRaises(ValueError):q.associate(*a)
 def test_sha_mismatch_refused(self):
  a=self.fixture();a[10]='b'*64
  with self.assertRaises(ValueError):q.associate(*a)
 def test_foreign_longer_covering_mount_refused(self):
  a=self.fixture();text=a[5]['original_line']+'\n712 711 0:43 /foreign /pti/lib ro - fuse.mergerfs 1:2 rw\n';a[5]=q.covering_mount(text,a[7]);a[6]=a[5]
  with self.assertRaises(ValueError):q.associate(*a)
if __name__=='__main__':unittest.main()
