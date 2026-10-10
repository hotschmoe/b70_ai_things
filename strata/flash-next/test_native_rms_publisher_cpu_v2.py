"""Synthetic original publisher4/page records; no real model/GPU reading."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import native_rms_publisher_binding_v2 as p
class Publisher(unittest.TestCase):
 def fixture(self,root):
  files=[];rows=[]
  for i in range(4):
   name='UD-Q4_K_XL/shard'+str(i)+'.gguf';path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(('CPU_SYNTHETIC_'+str(i)).encode());digest=p.sha(path);files.append({'path':name,'sha256':digest,'size':path.stat().st_size});rows.append({'path':str(path),'passed':True,'bytes':path.stat().st_size,'sha256':digest,'expected_sha256':digest,'stat_before':p.stat(path),'stat_after':p.stat(path)})
  lock={'files':files,'destination':'.','revision':'CPU_SYNTHETIC'};guard={'passed':True,'path':str(root/files[2]['path']),'stat_before':rows[2]['stat_after'],'stat_after':rows[2]['stat_after'],'rows':[],'epoch':15.}
  for off in (0,4096):
   file=root/('page-'+str(off)+'.raw');file.write_bytes(b'X'*4096);guard['rows'].append({'offset':off,'passed':True,'bytes':4096,'sha256':p.sha(file),'expected_sha256':p.sha(file),'preserved_path':str(file)})
  after=copy.deepcopy(guard);after['epoch']=21.;r={'post_full4':{'passed':True,'lock_sha256':'CPU_LOCK','model_revision':'CPU_SYNTHETIC','rows':rows,'started':16.,'finished':20.,'after_terminal_and_post_health_epoch':14.},'GPU_terminal_epoch':10.,'post_health':{'finished_epoch':12.},'post_journal':{'finished_epoch':14.},'finished_epoch':22.,'before_post_full4_pages':guard,'post_pages':after};return lock,r
 def bind(self,root,lock,row):
  with patch.object(p,'KNOWN_PAGES',[(0,hashlib.sha256(b'X'*4096).hexdigest()),(4096,hashlib.sha256(b'X'*4096).hexdigest())]):return p.publisher_binding(root,row,lock,'CPU_LOCK',root)
 def test_complete_ordered_publisher(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);lock,row=self.fixture(root);self.assertTrue(self.bind(root,lock,row)['complete_ordered_publisher4_current'])
 def test_size_hash_expected_roster_and_order_negatives(self):
  for kind in ('bytes','sha256','expected_sha256','order','missing'):
   with tempfile.TemporaryDirectory() as t:
    root=Path(t);lock,row=self.fixture(root)
    if kind=='order':row['post_full4']['rows'][0],row['post_full4']['rows'][1]=row['post_full4']['rows'][1],row['post_full4']['rows'][0]
    elif kind=='missing':row['post_full4']['rows'].pop()
    else:row['post_full4']['rows'][0][kind]=1 if kind=='bytes' else 'CPU_BAD'
    with self.assertRaises(ValueError):self.bind(root,lock,row)
 def test_terminal_journal_and_page_brackets(self):
  for kind in ('boundary','earlyScan','earlyAfter','lateBefore','offset','pagebytes'):
   with tempfile.TemporaryDirectory() as t:
    root=Path(t);lock,row=self.fixture(root)
    if kind=='boundary':row['post_full4']['after_terminal_and_post_health_epoch']=10.
    elif kind=='earlyScan':row['post_full4']['started']=13.
    elif kind=='earlyAfter':row['post_pages']['epoch']=19.
    elif kind=='lateBefore':row['before_post_full4_pages']['epoch']=17.
    elif kind=='offset':row['post_pages']['rows'][0]['offset']=8
    else:row['post_pages']['rows'][0]['bytes']=1
    with self.assertRaises(ValueError):self.bind(root,lock,row)
 def test_cache_profile_and_preservedV1(self):
  import qualify_native_rms_rsqrt37_v2 as q
  h=Path(__file__).parent;self.assertEqual(p.sha(h/'native-rms-rsqrt37-owned-source-plan-v1.json'),'7ee7968d44c030d8c99ccd72acf3f8c460736a8e446859c86ab41299be1b4354');self.assertIn("'SYCL_CACHE_PERSISTENT':'0'",Path(q.__file__).read_text())
if __name__=='__main__':unittest.main()
