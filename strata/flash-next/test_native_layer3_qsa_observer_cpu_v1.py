"""Source/synthetic contract checks; no C++/device/model execution."""
import copy,json,os,tempfile,unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import native_layer3_qsa_observer_contract_v1 as c
import prepare_native_layer3_qsa_observer_v1 as p

class Controls(unittest.TestCase):
 def record_all(self,record,rows):
  record.begin_capture(rows)
  if record.owner:
   for name,field in c.fields().items():record.copy(name,0,rows if field['rows']else 1)
  record.seal()
 def test_default_off_zero_object_effects_and_nonowner_zero_allocation(self):
  allocations=[]
  self.assertIsNone(c.Recorder.create(False,True,lambda:allocations.append(1)))
  nonowner=c.Recorder.create(True,False,lambda:allocations.append(1))
  self.assertEqual(allocations,[]);self.assertIsNone(nonowner.resource)
  self.record_all(nonowner,1)
  with self.assertRaises(ValueError):nonowner.copy('hc_mixed',0,1)
 def test_actual_shaped_2_1_1_nonce_protocol_and_stale_replay(self):
  owner=c.Recorder.create(True,True,lambda:object());other=c.Recorder.create(True,False,lambda:object())
  for route,pos,rows in c.WINDOWS:
   self.record_all(owner,rows);self.record_all(other,rows)
   owner.begin(123,1,0,route,pos,rows);other.begin(123,1,1,route,pos,rows)
   with self.assertRaises(ValueError):owner.returned(c.nonce(123,2,0,pos,rows))
   owner.returned(c.nonce(123,1,0,pos,rows));other.returned(None)
   with self.assertRaises(ValueError):owner.returned(c.nonce(123,1,0,pos,rows))
 def test_graph_row_roster_omission_duplicate_and_unsealed_refusal(self):
  owner=c.Recorder.create(True,True,lambda:object());owner.begin_capture(2)
  owner.copy('hc_mixed',0,1)
  with self.assertRaises(ValueError):owner.copy('hc_mixed',0,1)
  with self.assertRaises(ValueError):owner.seal()
  owner.begin(123,1,0,'prompt_verifier',0,2)
  with self.assertRaises(ValueError):owner.returned(c.nonce(123,1,0,0,2))
 def frame(self,stage=0,route='prompt_verifier',pos=0,rows=2):
  owner=stage==0;lb,le=((0,32)if owner else(32,48))
  return {'schema':1,'pid':123,'request':1,'stage':stage,'lb':lb,'le':le,'layer':3,'owner':owner,
   'route':route,'first_position':pos,'rows':rows,'nonce':c.nonce(123,1,stage,pos,rows),
   'device_nonce_observed':owner,'binding_sha256':'a'*64,'n_pages':512 if owner else 0,
   'n_slots':512 if owner else 0,'max_cells':2048 if owner else 0,
   'indexer_before_active_pooled_rows':pos//4+1 if pos else 0,
   'indexer_after_active_pooled_rows':(pos+rows)//4+1,
   'unused_padding_is_native_math_target':False,'forward_state_is_committed_state':False,
   'internal_norm_argument_or_rsqrt_observed':False,'internal_attention_score_softmax_observed':False,
   'captures_are_math_inputs':False,'full_model_math_qualified':False,
   'fields':[{'name':name,'bytes':size,'encoding':c.fields()[name]['encoding']}for name,size in c.quota(owner,rows).items()]}
 def test_owner_and_nonowner_full_three_window_rosters(self):
  for stage in(0,1):
   for route,pos,rows in c.WINDOWS:
    out=c.admit(self.frame(stage,route,pos,rows),'a'*64)
    self.assertTrue(out['capture_targets_only']);self.assertFalse(out['native_math_or_logical_pool_view_qualified'])
  self.assertEqual(len(c.fields()),29);self.assertEqual(c.quota(False,2),{})
 def test_complete_two_stage_and_one_stage_coverage_without_borrowing(self):
  frames=[self.frame(stage,route,pos,rows)for stage in(0,1)for route,pos,rows in c.WINDOWS]
  self.assertTrue(c.whole_prefix(frames,'a'*64,True)['stage1_nonowner_zero_quota_explicit'])
  with self.assertRaises(ValueError):c.whole_prefix(frames[:-1],'a'*64,True)
  bad=copy.deepcopy(frames);bad[-1]['request']=2;bad[-1]['nonce']=c.nonce(123,2,1,3,1)
  with self.assertRaises(ValueError):c.whole_prefix(bad,'a'*64,True)
  one=copy.deepcopy(frames[:3])
  for row in one:row['le']=48
  c.whole_prefix(one,'a'*64,False)
 def test_missing_duplicate_extent_scope_and_nonce_refused(self):
  source=self.frame()
  mutations=[lambda x:x['fields'].pop(),lambda x:x['fields'].append(x['fields'][0]),
   lambda x:x['fields'][0].update(bytes=1),lambda x:x.update(nonce=x['nonce']+1),
   lambda x:x.update(captures_are_math_inputs=True),lambda x:x.update(forward_state_is_committed_state=True),
   lambda x:x.update(internal_norm_argument_or_rsqrt_observed=True),lambda x:x.update(stage=1)]
  for change in mutations:
   value=copy.deepcopy(source);change(value)
   with self.assertRaises(ValueError):c.admit(value,'a'*64)
 def test_wrong_window_model_binding_and_pool_map_refused(self):
  for changed in({'first_position':1},{'binding_sha256':'b'*64},{'indexer_after_active_pooled_rows':2},{'n_slots':1}):
   value=self.frame();value.update(changed)
   with self.assertRaises(ValueError):c.admit(value,'a'*64)
  self.assertFalse(c.logical_page0(0,0)['identity_mapping_inferred_from_geometry'])
  for before,after in((1,1),(0,1),(-1,0),(True,0)):
   with self.assertRaises(ValueError):c.logical_page0(before,after)
 def test_future_collector_real_shaped_raw_step_map_and_active_width(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);frame=self.frame();frame_path=root/'frame.json'
   for item in frame['fields']:
    item['file']=item['name']+'.bin';(root/item['file']).write_bytes(bytes(item['bytes']))
   (root/'step.bin').write_bytes(np.asarray([[0,1,0,1],[1,2,0,2]],dtype='<i4').tobytes())
   (root/'selected_ids_capacity4.bin').write_bytes(np.asarray([[0,-99,-99,-99],[0,1,-99,-99]],dtype='<i4').tobytes())
   frame_path.write_text(json.dumps(frame));self.assertTrue(c.recollect(frame_path,'a'*64)['physical_page0_logical_view_admitted'])
   for name,raw in [('page_before',np.asarray([1],dtype='<i4').tobytes()),
                    ('page_resolved',np.asarray([-1],dtype='<i4').tobytes()),
                    ('step',np.asarray([[0,1,0,5],[1,2,0,2]],dtype='<i4').tobytes()),
                    ('selected_ids_capacity4',np.asarray([[1,-99,-99,-99],[0,1,-99,-99]],dtype='<i4').tobytes())]:
    file=root/(name+'.bin');original=file.read_bytes();file.write_bytes(raw)
    with self.assertRaises(ValueError):c.recollect(frame_path,'a'*64)
    file.write_bytes(original)
 def test_descriptor_failure_cleanup_and_release_idempotence(self):
  with tempfile.TemporaryDirectory()as directory:
   recorded=[];real=os.open
   def observe(*args):fd=real(*args);recorded.append(fd);return fd
   with patch.object(c.os,'open',side_effect=observe):
    with self.assertRaises(MemoryError):c.FailureModel(directory,lambda:(_ for _ in()).throw(MemoryError('synthetic')))
   with self.assertRaises(OSError):os.fstat(recorded[0])
   model=c.FailureModel(directory,lambda:object());freed=[]
   model.release(lambda r:freed.append(r));model.close(lambda r:freed.append(r))
   self.assertEqual(len(freed),1);self.assertEqual(model.success_markers,1)
 def test_failed_free_never_emits_success(self):
  with tempfile.TemporaryDirectory()as directory:
   model=c.FailureModel(directory,lambda:object())
   with self.assertRaises(RuntimeError):model.close(lambda r:(_ for _ in()).throw(RuntimeError('synthetic')))
   self.assertEqual(model.success_markers,0);self.assertFalse(model.released);self.assertEqual(model.fd,-1)
 def test_new_plan_patch_closure_and_unchanged_math(self):
  old,new=p.reconstruct();plan=p.build_plan()
  self.assertEqual(p.PATCH.read_bytes(),p.patch_bytes());self.assertEqual(len(plan['patches']),40)
  self.assertEqual((len(plan['expected_patched_source_sha256']),len(plan['added_header_payloads'])),(67,31))
  self.assertEqual(set(n for n in new if old.get(n)!=new[n]),{p.H,'sycl/include/strata/core/verify.hpp','sycl/src/core/verify.cpp'})
  for name,text in new.items():self.assertEqual(p.sha_bytes(text)if hasattr(p,'sha_bytes')else __import__('hashlib').sha256(text.encode()).hexdigest(),plan['expected_patched_source_sha256'][name])
  source=new['sycl/src/core/verify.cpp']
  for line in old['sycl/src/core/verify.cpp'].splitlines():self.assertIn(line,source)
  self.assertIn('copy("mixed_q81",xq_,tb,n,T',source);self.assertIn('copy("gated_q81",xq_,tb,n,T',source)
  self.assertIn('qsa3_snapshot_.reset(); // P30/fidelity graphs retired',source)
 def test_default_off_and_constructor_lifetime_source_contract(self):
  header=p.HEADER.read_text();self.assertIn('if(!enabled)return;',header)
  self.assertIn('if(released_)return;',header);self.assertIn('catch(...){::close(fd_);fd_=-1;throw;}',header)
  self.assertIn('internal_norm_argument_or_rsqrt_observed\\\":false',header)
  self.assertNotIn('sycl::rsqrt(',header);self.assertNotIn('sycl::fma(',header)
  source=p.reconstruct()[1]['sycl/src/core/verify.cpp']
  self.assertIn('if(qsa3_target::settings().enabled)',source)
  self.assertIn('qsa3_snapshot_&&(prefix30_prompt_||fidelity_first_)&&l==3',source)
 def test_exact_corrected_header_payload_embedded_in_patch(self):
  text=p.PATCH.read_text();start='+++ b/'+p.H+'\n';self.assertEqual(text.count(start),1)
  remaining=text.split(start,1)[1];end=remaining.find('\n--- ')
  section=remaining if end<0 else remaining[:end+1]
  payload=''.join(line[1:] for line in section.splitlines(True)if line.startswith('+'))
  self.assertEqual(payload,p.HEADER.read_text())
  self.assertIn('if(released_)return;',payload)
  self.assertIn('catch(...){::close(fd_);fd_=-1;throw;}',payload)
 def test_source_plan_derived_separately_from_frozen39(self):
  actual=json.loads((p.HERE/'native-layer3-qsa-observer-engine-build-plan-v1.json').read_text())
  self.assertEqual(actual,p.build_plan());self.assertEqual(p.sha(p.BASE),p.BASE_SHA)
  self.assertFalse(actual['new_source_contracts']['0040']['captured_values_are_math_inputs'])
  self.assertFalse(actual['new_source_contracts']['0040']['source39_runtime_transfer'])

if __name__=='__main__':unittest.main()
