import copy,unittest
from full_cache_shared_memory_v2 import host_observation

class Memory(unittest.TestCase):
 def sample(self):return {'kind':'owned_host_memory','host_controller_pid':17,'container_id':'owned','cgroup_memory_current_bytes':8,'cgroup_memory_peak_bytes':10,'cgroup_anon_bytes':2,'cgroup_file_bytes':4,'cgroup_shmem_bytes':2,'cgroup_swap_current_bytes':0,'whole_system_physical_peak_qualified':False,'per_expert_device_residency_qualified':False,'raw_receipts':[{'sha256':'a'*64,'bytes':10}]}
 def test_scope_not_physical_or_expert_claim(self):
  result=host_observation(self.sample());self.assertEqual(result['cgroup_sampled_peak_bytes'],10);self.assertFalse(result['per_expert_device_residency_qualified']);self.assertTrue(result['original_raw_receipts_still_need_owned_recollection'])
 def test_swap_type_peak_and_physical_flags_rejected(self):
  for key,value in [('cgroup_memory_current_bytes',True),('cgroup_memory_peak_bytes',7),('cgroup_swap_current_bytes',1),('per_expert_device_residency_qualified',True),('whole_system_physical_peak_qualified',True),('raw_receipts',[])]:
   sample=self.sample();sample[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):host_observation(sample)

if __name__=='__main__':unittest.main()
