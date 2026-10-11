import json,re,subprocess,tempfile,unittest
from pathlib import Path
import prepare_gdn_core44_capture_v1 as p
class Controls(unittest.TestCase):
 def test_real_patch_apply_and_exact_undo_five_files(self):
  before=p.originals();after=p.changes()
  with tempfile.TemporaryDirectory()as t:
   root=Path(t)
   for n,b in before.items():f=root/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
   patch=p.patch_text().encode();self.assertEqual(subprocess.run(['patch','--batch','-p1'],cwd=root,input=patch,capture_output=True).returncode,0)
   for n,b in after.items():self.assertEqual((root/n).read_bytes(),b)
   self.assertEqual(subprocess.run(['patch','--batch','-R','-p1'],cwd=root,input=patch,capture_output=True).returncode,0)
   for n,b in before.items():self.assertEqual((root/n).read_bytes(),b)
 def test_complete_bounded_38_fields_and_original_prefix_unchanged(self):
  old=p.originals()[p.CONTRACT].decode();new=p.changes()[p.CONTRACT].decode();pattern=r'\{"([^"]+)","([^"]+)",(\d+)\}';a=re.findall(pattern,old);b=re.findall(pattern,new);self.assertEqual(len(a),33);self.assertEqual(len(b),38);self.assertEqual(b[:33],a);self.assertEqual([(n,e,int(s))for n,e,s in b[33:]],list(p.FIELDS));self.assertEqual(sum(s for _,_,s in p.FIELDS),25156);self.assertLess(sum(int(s)for _,_,s in b)*4,64<<20)
 def test_original_host_dispatch_entry_byte_unchanged(self):
  old=p.originals()[p.KERNEL].decode();new=p.changes()[p.KERNEL].decode().replace(p.LAUNCH,'');begin='void gdn_step_norm_multi(float*';end='\nnamespace {\n__dpct_inline__ void wait_flag_ge_kernel';self.assertEqual(old[old.index(begin):old.index(end,old.index(begin))],new[new.index(begin):new.index(end,new.index(begin))])
 def test_no_new_query_barrier_or_kernel_node_in_old_dispatch(self):
  old=p.originals()[p.KERNEL].decode();new=p.changes()[p.KERNEL].decode()
  for name in('item_ct1.barrier();','zeDeviceGetGlobalTimestamps','wait_and_throw','sycl::atomic_fence'):
   self.assertEqual(new.count(name),old.count(name))
  self.assertEqual(new.count('->parallel_for<'),old.count('->parallel_for<')+1);self.assertIn('gdn_step_norm_multi_kernel<false,false,true>',p.LAUNCH);self.assertIn('n_tok!=1',p.LAUNCH);self.assertIn('!observed.error',p.LAUNCH)
 def test_math_expression_source_and_old_false_specialization(self):
  k=p.changes()[p.KERNEL].decode();self.assertIn('bool CAPTURE44=false',k);self.assertIn('sycl::rsqrt(ss / (float)S + eps)',k);self.assertIn('observed.argument[head]=argument',k);self.assertIn('observed.scale[head]=scale',k);self.assertIn('observed.square_sum[head]=ss',k);self.assertIn('observed.core[head*S+col]=oc',k);self.assertIn('if(head==0&&col==0)observed.inverse_s[0]=inverse_s',k);self.assertIn('*observed.error|=4096u',k)
 def test_snapshot_owned_ranges_roster_nonce_and_release_retained(self):
  old=p.originals()[p.OBSERVER].decode();new=p.changes()[p.OBSERVER].decode()
  for name,_,size in p.FIELDS:self.assertIn('target("%s",%d)'%(name,size),new)
  self.assertIn('rosters_[building_][i]=true',new);self.assertIn('affinity(q)',p.DEVICE_TARGETS)
  for fragment in('q_.wait_and_throw();','sycl::free(data_,q_);','coverage[0]','published_==nonce_','key!=armed_key_'):self.assertEqual(new.count(fragment),old.count(fragment))
 def test_OFF_uses_original_call_and_only_l0_uses_new_entry(self):
  v=p.changes()[p.VERIFY].decode();self.assertEqual(v.count('gdn_step_norm_observed44('),1);self.assertIn('if(l0_capture)gdn_step_norm_observed44',v);self.assertIn('else\n                gdn_step_norm_multi(state',v);self.assertIn('STRATA_LAYER0_Q8_DIAG',p.originals()[p.OBSERVER].decode())
 def test_fresh_build_proposal_preserves_old_43_and_requires_all8(self):
  with tempfile.TemporaryDirectory(dir=p.ROOT)as t:
   patch=Path(t)/'patch';patch.write_text(p.patch_text());value=p.proposal(patch);old=json.loads(p.PARENT.read_bytes());self.assertEqual(value['patches'][:-1],old['patches']);self.assertEqual(value['source_generation'],44);self.assertEqual(len(value['overlay_files']),71);self.assertEqual(len(value['expected_patched_source_sha256']),71);self.assertEqual(value['runtime_python_sources'],old['runtime_python_sources']);self.assertEqual(value['build_targets'],old['build_targets']);self.assertEqual(len(value['build_targets']),8);self.assertFalse(value['source43_runtime_proof_transferred']);self.assertEqual(value['source44_before_patch_inputs'],{n:p.sha(b)for n,b in p.originals().items()})
if __name__=='__main__':unittest.main()
