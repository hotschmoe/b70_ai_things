import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_memory_capture_v2 as m

class Capture(unittest.TestCase):
 def stat(self,pid,parent,state='S'):
  return (str(pid)+' (owned worker) '+state+' '+str(parent)+' '+' '.join(['0']*17+['123'])+'\n').encode()
 def fixture(self,root):
  actor=root/'actor';actor.mkdir();plan={'image':'pinned'};(actor/'plan.snapshot.json').write_text(json.dumps(plan));sampledir=actor/'memory-samples'/'sample0';sampledir.mkdir(parents=True)
  digest=hashlib.sha256((actor/'plan.snapshot.json').read_bytes()).hexdigest();command=['docker','run','--name','b70-prefix-17-worker','--user','1000:1000','--group-add','109','--group-add','44','-v','/source:/src:ro','-e','EXPLICIT=1','pinned','-lc','original command'];obj={'Id':'actual','Name':'/b70-prefix-17-worker','Image':'pinned','Mounts':[{'Source':'/source','Destination':'/src','RW':False,'Type':'bind'}],'HostConfig':{'NetworkMode':'host','Memory':105*1024**3,'MemorySwap':105*1024**3,'Privileged':False,'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'GroupAdd':['109','44'],'Binds':['/source:/src:ro']},'Config':{'Image':'pinned','Cmd':['-lc','original command'],'Entrypoint':['/bin/bash'],'User':'1000:1000','Env':['EXPLICIT=1'],'Labels':{'b70.prefix.plan':digest}},'State':{'Pid':101,'Running':True}}
  raw={'root-cgroup':b'0::/owned\n','stat-101':self.stat(101,17),'tasks-101':b'101 111','children-101-101':b'','children-101-111':b'102','stat-102':self.stat(102,101),'tasks-102':b'102','children-102-102':b'','status-101':b'Pid:101','status-102':b'Pid:102','smaps-101':b'Rss:4 kB','smaps-102':b'Rss:2 kB','fdinfo-101-4':b'drm-memory-vram: 8 KiB'}
  vals={'current':8,'peak':10,'max':105*1024**3,'swap-current':0,'swap-max':0};raw.update({'cgroup-memory-'+k:str(v).encode() for k,v in vals.items()});raw['cgroup-memory-stat']=b'anon 2\nfile 4\nshmem 2\n';raw['cgroup-memory-events']=b'oom 0\n'
  receipts=[]
  for name,value in raw.items():
   p=sampledir/(name+'.raw');p.write_bytes(value);receipts.append({'path':str(p),'kernel_path':'/fake/'+name,'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value),'started_epoch':1.,'finished_epoch':2.})
  sample={'started_epoch':0.,'finished_epoch':3.,'kind':'owned_host_memory','host_controller_pid':17,'container_id':'actual','container_root_host_pid':101,'owned_process_identities':[m.parse_stat(raw['stat-101'],101),m.parse_stat(raw['stat-102'],102)],'actor_plan_sha256':digest,'actor_command_sha256':hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest(),'cgroup_path':'/owned','cgroup_memory_current_bytes':8,'cgroup_memory_peak_bytes':10,'cgroup_anon_bytes':2,'cgroup_file_bytes':4,'cgroup_shmem_bytes':2,'cgroup_swap_current_bytes':0,'whole_system_physical_peak_qualified':False,'per_expert_device_residency_qualified':False,'drm_fdinfo_raw_observed':True,'raw_receipts':receipts,'ownership_receipt':{'inspection':obj,'command':command,'actor_root':str(actor)}}
  return sampledir,sample,command
 def save(self,root,sample):(root/'sample.json').write_text(json.dumps(sample))
 def test_raw_recollection_owned_tree_and_scope(self):
  with tempfile.TemporaryDirectory() as td:
   root,sample,command=self.fixture(Path(td));self.save(root,sample)
   with patch('run_full_cache_shared_runtime_v2.command_recipe',return_value=command):result=m.recollect(root,sample)
   self.assertFalse(result['original_raw_receipts_still_need_owned_recollection']);self.assertFalse(result['per_expert_device_residency_qualified'])
 def test_mutated_raw_even_with_updated_receipt_rejects_ancestry(self):
  with tempfile.TemporaryDirectory() as td:
   root,sample,command=self.fixture(Path(td));p=root/'stat-102.raw';p.write_bytes(self.stat(102,99))
   row=next(r for r in sample['raw_receipts'] if r['path']==str(p));row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=len(p.read_bytes()));self.save(root,sample)
   with patch('run_full_cache_shared_runtime_v2.command_recipe',return_value=command),self.assertRaises(ValueError):m.recollect(root,sample)
 def test_swap_limit_and_foreign_owner_rejected(self):
  for mutate in ('swap','owner'):
   with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as td:
    root,sample,command=self.fixture(Path(td))
    if mutate=='owner':sample['ownership_receipt']['inspection']['Id']='foreign'
    else:
     p=root/'cgroup-memory-swap-max.raw';p.write_bytes(b'1');row=next(r for r in sample['raw_receipts'] if r['path']==str(p));row['sha256']=hashlib.sha256(b'1').hexdigest()
    self.save(root,sample)
    with patch('run_full_cache_shared_runtime_v2.command_recipe',return_value=command),self.assertRaises(ValueError):m.recollect(root,sample)
 def test_actual_recipe_mutations_rejected(self):
  import copy
  with tempfile.TemporaryDirectory() as td:
   root,sample,command=self.fixture(Path(td));original=sample['ownership_receipt']['inspection']
   for area,key,value in [('Config','Cmd',['foreign']),('Config','User','0'),('HostConfig','Memory',1),('HostConfig','Devices',[]),('Config','Env',[])]:
    with self.subTest(key=key):
     obj=copy.deepcopy(original);obj[area][key]=value
     with self.assertRaises(ValueError):m.recipe_binding(obj,command,'pinned')
 def test_transient_process_state_not_identity_change(self):
  self.assertTrue(m.stable(m.parse_stat(self.stat(101,17,'R'),101),m.parse_stat(self.stat(101,17,'S'),101)));self.assertFalse(m.stable(m.parse_stat(self.stat(101,17),101),m.parse_stat(self.stat(101,17,'Z'),101)))
if __name__=='__main__':unittest.main()
