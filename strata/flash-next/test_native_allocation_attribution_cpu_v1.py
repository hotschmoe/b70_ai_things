"""Synthetic event replay and exact draft hook contract; no native execution."""
import copy,json,tempfile,unittest
from pathlib import Path
import native_allocation_attribution_contract_v1 as c
import native_allocation_attribution_reader_v1 as r
import prepare_native_allocation_attribution_v1 as p
class Controls(unittest.TestCase):
 def owner(self,host=False):return {'engine_pid':17,'process_start_ticks':123,'device_uuid':None if host else '1'*32,'source_generation':1000000}
 def rows(self,host=False):
  owner=self.owner(host);a={'owner':owner,'sequence':1,'host_epoch':1.0,'allocation_id':'17-123-1','stage':-1 if host else 0,'owner_generation':1,'queue_context':'host_process' if host else 'ctx_0x1','role':'host_cache_vector' if host else 'expert_cache_arena','space':'host_heap' if host else 'device_usm','api_result':'success','bytes':128,'offset':0,'layer':-1,'expert':-1,'copy_completion_observed':False,'existing_completion_boundary':False,'completion_boundary':'none','command_context':{'current_main_body_rid':0,'actual_slot_rid':0,'phase':'startup'},'actual_slot_context_observed':False,'physical_residency_observed':False,'kind':'allocation_success'}
  rows=[a]
  if not host:
   rows.append(dict(a,sequence=2,host_epoch=2.0,kind='expert_bytes_ready',layer=3,expert=7,bytes=64,copy_completion_observed=True,existing_completion_boundary=True,completion_boundary='blocking_copy_wait_returned'));rows.append(dict(rows[-1],sequence=3,host_epoch=3.0,kind='expert_bytes_released',copy_completion_observed=False,existing_completion_boundary=False,completion_boundary='owning_free_returned'))
  rows.append(dict(a,sequence=len(rows)+1,host_epoch=float(len(rows)+1),kind='allocation_release_success'));return rows
 def test_complete_backed_USM_and_host_heap_payload_scopes_without_residency_claim(self):
  d=c.analyze(self.rows(),self.owner(),[[3,7]],1000000);h=c.analyze(self.rows(True),self.owner(True),[],1000000);self.assertEqual(d['peak_live_API_bytes_by_space']['device_usm'],128);self.assertEqual(h['peak_live_API_bytes_by_space']['host_heap'],128);self.assertTrue(h['actual_host_cache_payload_API_bytes_observed']);self.assertFalse(h['physical_residency_or_eviction_qualified']);self.assertFalse(h['whole_host_cache_metadata_allocations_qualified'])
 def test_pending_copy_added_barrier_and_unlogged_free_refused(self):
  for key,value in [('copy_completion_observed',False),('existing_completion_boundary',False),('completion_boundary','diagnostic_added_wait')]:
   rows=self.rows();rows[1][key]=value;self.assertRaises(ValueError,c.analyze,rows,self.owner(),[[3,7]],1000000)
  self.assertRaises(ValueError,c.analyze,self.rows()[:-1],self.owner(),[[3,7]],1000000)
 def test_typed_owner_stage_and_release_range_mutations_refused(self):
  for index,key,value in [(1,'stage',False),(1,'owner_generation',2),(2,'offset',True),(2,'bytes',63),(3,'bytes',127),(1,'space','virtual_address')]:
   rows=self.rows();rows[index][key]=value;self.assertRaises(ValueError,c.analyze,rows,self.owner(),[[3,7]],1000000)
  rows=self.rows();rows[1]['owner']=dict(self.owner(),source_generation=1000000.0);self.assertRaises(ValueError,c.analyze,rows,self.owner(),[[3,7]],1000000)
 def test_wrong_empty_or_extra_expert_roster_refused(self):
  for expected in ([],[[3,8]],[[3,7],[3,8]],[[3,7],[3,7]]):self.assertRaises(ValueError,c.analyze,self.rows(),self.owner(),expected,1000000)
 def test_virtual_handle_host_mapping_and_host_domain_expert_refused(self):
  for space in ('virtual_address','physical_handle','host_mapping','host_heap'):
   rows=self.rows();rows[0]['space']=space;self.assertRaises(ValueError,c.analyze,rows,self.owner(),[[3,7]],1000000)
  self.assertRaises(ValueError,c.analyze,self.rows(True),self.owner(True),[[3,7]],1000000)
 def test_complete_bounded_stream_and_missing_duplicate_truncated_error_refused(self):
  owners=[{'owner':self.owner(),'expected_experts':[[3,7]],'expected_roles':['expert_cache_arena'],'expected_stages':[0]},{'owner':self.owner(True),'expected_experts':[],'expected_roles':['host_cache_vector'],'expected_stages':[-1]}];text=''.join('ALLOC_ATTR '+json.dumps(row)+'\n' for row in self.rows()+self.rows(True))
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'events.log';path.write_text(text);self.assertEqual(len(r.recollect(path,owners,1000000)['scoped_owner_reports']),2)
   for changed in (text.rstrip('\n'),text+'ALLOC_ATTR_ERROR failure\n',text.replace('"kind": "allocation_success"','"kind": "allocation_success", "kind":"allocation_success"',1),''.join('ALLOC_ATTR '+json.dumps(row)+'\n' for row in self.rows())):
    path.write_text(changed);self.assertRaises(ValueError,r.recollect,path,owners,1000000)
 def test_defaultOFF_no_query_wait_or_allocation_in_helper_fast_path(self):
  h=(p.HERE/'native_allocation_attribution_sycl_v1.hpp').read_text();self.assertLess(h.index('if(!enabled())return'),h.index('sycl::get_pointer_type'));self.assertLess(h.index('if(!enabled())return'),h.index('zeDeviceGetProperties'));self.assertNotIn('wait_and_throw',h);self.assertNotIn('malloc_',h)
 def test_emitter_single_locked_full_record_and_actual_start_identity(self):
  h=(p.HERE/'native_allocation_attribution_producer_v1.hpp').read_text();emit=h[h.index('inline void emit('):h.index('inline void allocation_success')];self.assertEqual(emit.count('std::fwrite('),1);self.assertIn('flockfile(stderr)',emit);self.assertIn('funlockfile(stderr)',emit);self.assertNotIn('std::fprintf(',emit);self.assertIn('bool slot_observed=false',h);self.assertIn('/proc/self/stat',h)
 def test_reconstructed_patch_exact_and_original_math_source_untouched(self):
  old,new=p.reconstruct();self.assertEqual(p.patch_bytes(),p.PATCH.read_bytes());changed={name for name in new if old.get(name)!=new[name]};self.assertEqual(len(changed),13);self.assertFalse(any('/kernels/' in name for name in changed));self.assertEqual(old['sycl/src/core/session.cpp'],new['sycl/src/core/session.cpp']);self.assertEqual(old['sycl/src/core/verify.cpp'],new['sycl/src/core/verify.cpp'])
 def test_actual_existing_wait_and_read_worker_boundaries_before_ready_hooks(self):
  _,new=p.reconstruct();cache=new['sycl/src/core/expert_cache.cpp']
  for name in ('fill_slot_blocking','fill_slot_queued'):
   block=cache[cache.index('bool ExpertCache::'+name+'('):];block=block[:block.index('\ncatch (sycl::exception')];self.assertLess(block.index('.wait()'),block.index('allocation_observer_ready'))
  mirror=new['sycl/src/core/gguf_expert_source.cpp'];self.assertLess(mirror.index('for (auto& thread:ts) thread.join();'),mirror.index('allocation_attribution::usm_ready'));self.assertLess(mirror.index('if (bad) { err="stage mirror'),mirror.index('allocation_attribution::usm_ready'))
 def test_successful_USM_release_and_host_allocator_API_hooks(self):
  _,new=p.reconstruct();mirror=new['sycl/include/strata/core/stage_expert_mirror.hpp'];self.assertLess(mirror.index('sycl::free(pointer,q)'),mirror.index('allocation_attribution::release_returned'));self.assertIn('allocation_key=allocation_attribution::usm_key(pointer,q)',mirror);self.assertIn('allocation_attribution::host_vector<uint8_t> gdn, ple, tails, dead, block_pos',new['include/strata/core/conversation_cache.hpp']);self.assertIn('allocation_attribution::host_vector<Segment> segments_',new['include/strata/core/conversation_buffer.hpp']);self.assertIn('template<class T,class A> void vec',new['src/core/conversation_file.cpp'])
if __name__=='__main__':unittest.main()
