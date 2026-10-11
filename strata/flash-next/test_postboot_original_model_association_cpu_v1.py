import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import postboot_original_model_association_v1 as p
class Controls(unittest.TestCase):
 def fixture(self,root):
  lock={'destination':'CPU_MODELS','revision':'original-revision','files':[{'path':'UD-Q4_K_XL/shard'+str(i),'size':10+i,'sha256':str(i)*64}for i in range(4)]};rows=[]
  for i,want in enumerate(lock['files']):rows.append({'path':str(p.ROOT/lock['destination']/want['path']),'expected_sha256':want['sha256'],'sha256':want['sha256'],'bytes':want['size'],'passed':True,'stat_before':[51,100+i,10+i,1000+i,2000+i],'stat_after':[51,100+i,10+i,1000+i,2000+i]})
  old={'passed':True,'lock_sha256':'a'*64,'model_revision':lock['revision'],'started':10.,'finished':20.,'rows':rows};new=copy.deepcopy(old);new['started']=30.;new['finished']=40.
  for row in new['rows']:row['stat_before'][0]=row['stat_after'][0]=45
  return lock,old,new
 def run_association(self,root,lock,old,new):
  a=root/'old.json';b=root/'current.json';a.write_text(json.dumps(old));b.write_text(json.dumps(new));actual=p.read_unique
  with patch.object(p,'read_unique',side_effect=lambda path:lock if Path(path)==p.HERE/'model-lock.json'else actual(path)),patch.object(p,'sha',return_value='a'*64):return p.association(a,b,current_expected_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),current_stat_reader=lambda path:next(r['stat_after']for r in new['rows']if r['path']==path))
 def test_exact_changed_device_preserves_original_reports_and_old_failure(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);lock,old,new=self.fixture(root);result=self.run_association(root,lock,old,new);self.assertFalse(result['old_current_stat_gate_passed']);self.assertFalse(result['historical_GPU_health_transferred']);self.assertTrue(result['original_bytes_requalified_by_actual_full4']);self.assertEqual(json.loads((root/'old.json').read_text()),old)
 def test_every_nondevice_stat_change_refused(self):
  for index in range(1,5):
   with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp);lock,old,new=self.fixture(root);new['rows'][0]['stat_before'][index]+=1;new['rows'][0]['stat_after'][index]+=1;self.assertRaises(ValueError,self.run_association,root,lock,old,new)
 def test_wrong_hash_order_count_or_device_mapping_refused(self):
  for mutation in('hash','order','missing','same_device','bool_stat'):
   with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp);lock,old,new=self.fixture(root)
    if mutation=='hash':new['rows'][0]['sha256']='f'*64
    elif mutation=='order':new['rows'].reverse()
    elif mutation=='missing':new['rows'].pop()
    elif mutation=='same_device':new['rows'][0]['stat_before'][0]=new['rows'][0]['stat_after'][0]=51
    else:new['rows'][0]['stat_before'][0]=new['rows'][0]['stat_after'][0]=True
    self.assertRaises(ValueError,self.run_association,root,lock,old,new)
if __name__=='__main__':unittest.main()
