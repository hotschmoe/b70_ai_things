"""Per-placement, tagged host, bounded consumed-log and source-hook controls."""
import copy,tempfile,json,hashlib,unittest
from pathlib import Path
import test_native_allocation_attribution_cpu_v1 as old
import native_allocation_attribution_contract_v2 as c
import native_allocation_attribution_reader_v2 as r
import prepare_native_allocation_attribution_v2 as p
class Controls(unittest.TestCase):
 def owner(self,host=False):return old.Controls().owner(host)
 def rows(self,host=False):
  rows=old.Controls().rows(host)
  for row in rows:row.update(role_index=0,element_size=1 if host else 0,element_count=128 if host else 0)
  if host:
   for row in rows:row['role']='checkpoint_payload'
  return rows
 def place(self,role='expert_cache_arena',context='ctx_0x1',index=0):return {'stage':0,'queue_context':context,'role':role,'role_index':index,'layer':3,'expert':7}
 def two(self):
  a=self.rows();b=copy.deepcopy(a)
  for row in b:row.update(allocation_id='17-123-2',owner_generation=2,role='stage_mirror_segment',role_index=1)
  rows=[a[0],b[0],a[1],b[1],a[2],b[2],a[3],b[3]]
  for i,row in enumerate(rows):row.update(sequence=i+1,host_epoch=float(i+1))
  return rows
 def test_legitimate_same_expert_two_backed_roles_requires_both_placements(self):
  x=c.analyze(self.two(),self.owner(),[self.place(),self.place('stage_mirror_segment',index=1)],1000000,[]);self.assertTrue(x['complete_declared_expert_placements_observed']);self.assertEqual(x['peak_live_API_bytes_by_space']['device_usm'],256)
 def test_one_ready_cannot_satisfy_second_declared_copy(self):
  rows=[row for row in self.two()if not(row['role']=='stage_mirror_segment' and row['kind'].startswith('expert_bytes'))]
  for i,row in enumerate(rows):row.update(sequence=i+1,host_epoch=float(i+1))
  self.assertRaises(ValueError,c.analyze,rows,self.owner(),[self.place(),self.place('stage_mirror_segment',index=1)],1000000,[])
 def test_context_stage_and_role_index_placement_borrowing_refused(self):
  for key,value in [('queue_context','ctx_foreign'),('stage',1),('role_index',2)]:
   wanted=self.place();wanted[key]=value;self.assertRaises(ValueError,c.analyze,self.rows(),self.owner(),[wanted],1000000,[])
 def test_directory_only_never_claims_payload_or_container_instance(self):
  rows=self.rows(True)
  for row in rows:row.update(role='conversation_directory',element_size=32,element_count=4)
  x=c.analyze(rows,self.owner(True),[],1000000,['conversation_directory']);self.assertTrue(x['tagged_host_vector_API_bytes_observed']);self.assertFalse(x['actual_host_cache_payload_API_bytes_observed']);self.assertFalse(x['owning_checkpoint_or_buffer_instance_observed'])
 def test_payload_coverage_requires_declared_both_tags_and_exact_element_extent(self):
  a=self.rows(True);b=copy.deepcopy(a)
  for row in b:row.update(role='conversation_payload',allocation_id='17-123-2',owner_generation=2)
  rows=[a[0],b[0],a[1],b[1]]
  for i,row in enumerate(rows):row.update(sequence=i+1,host_epoch=float(i+1))
  self.assertTrue(c.analyze(rows,self.owner(True),[],1000000,['checkpoint_payload','conversation_payload'])['actual_host_cache_payload_API_bytes_observed']);bad=copy.deepcopy(rows);bad[0]['element_count']=127;self.assertRaises(ValueError,c.analyze,bad,self.owner(True),[],1000000,['checkpoint_payload','conversation_payload'])
 def test_actual_original_log_SHA_and_complete_bounded_lines(self):
  rows=self.rows()+self.rows(True);owners=[{'owner':self.owner(),'expected_placements':[self.place()],'expected_host_roles':[],'expected_roles':['expert_cache_arena'],'expected_stages':[0]},{'owner':self.owner(True),'expected_placements':[],'expected_host_roles':['checkpoint_payload'],'expected_roles':['checkpoint_payload'],'expected_stages':[-1]}];raw=''.join('ALLOC_ATTR '+json.dumps(row)+'\n'for row in rows).encode()
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'log';path.write_bytes(raw);proof=r.recollect(path,owners,1000000,hashlib.sha256(raw).hexdigest());self.assertTrue(proof['original_log_binding']['consumed_bytes_directly_SHA_bound'])
   self.assertRaises(ValueError,r.recollect,path,owners,1000000,'0'*64)
   for changed in (raw.rstrip(b'\n'),b'ALLOC_ATTR '+b'x'*16385+b'\n'):
    path.write_bytes(changed);self.assertRaises(ValueError,r.recollect,path,owners,1000000,hashlib.sha256(changed).hexdigest())
 def test_failure_or_duplicate_release_and_wrong_type_refused(self):
  for index,key,value in [(1,'copy_completion_observed',False),(2,'offset',True),(3,'bytes',127),(1,'owner_generation',2)]:
   rows=self.rows();rows[index][key]=value;self.assertRaises(ValueError,c.analyze,rows,self.owner(),[self.place()],1000000,[])
 def test_free_callback_occurs_under_tracker_lock_before_success_record(self):
  h=(p.HERE/'native_allocation_attribution_producer_v2.hpp').read_text();block=h[h.index('template<class Free>inline void release_owned'):h.index('// Only cache vectors')];self.assertLess(block.index('lock(t.mutex)'),block.index('free_api();auto&'));self.assertLess(block.index('free_api();auto&'),block.index('emit(t,"allocation_release_success"'));self.assertIn('registration cannot pass this lock',block)
 def test_hooks_clear_actual_member_after_API_before_record_failure(self):
  _,new=p.reconstruct();mirror=new['sycl/include/strata/core/stage_expert_mirror.hpp'];self.assertIn('traced_free(Pointer*& pointer)',mirror);self.assertIn('release_owned(allocation_key,[&]{sycl::free(pointer,q);pointer=nullptr;})',mirror);self.assertIn('release_owned(allocation_key,[&]{sycl::free(base_,queue_);base_=nullptr;})',new['sycl/include/strata/core/slot_session_arena.hpp'])
 def test_exact_patch_and_distinct_source_bound_payload_directory_roles(self):
  oldtree,new=p.reconstruct();self.assertEqual(p.PATCH.read_bytes(),p.patch_bytes());self.assertIn('checkpoint_vector<uint8_t> gdn, ple',new['include/strata/core/conversation_cache.hpp']);self.assertIn('conversation_payload_vector<uint8_t>',new['include/strata/core/conversation_buffer.hpp']);self.assertIn('conversation_directory_vector<Segment>',new['include/strata/core/conversation_buffer.hpp']);self.assertEqual(oldtree['sycl/src/core/verify.cpp'],new['sycl/src/core/verify.cpp'])
if __name__=='__main__':unittest.main()
