"""Source-indexed roster controls only; no SYCL execution or model operands."""
import hashlib,json,re,tempfile,unittest
from pathlib import Path
import prepare_batch_stage_stamp37_v1 as p

class Roster:
 """CPU semantic witness of the exact observer.hpp producer/seal checks."""
 def __init__(self,lb,le,head,layout):self.lb=lb;self.le=le;self.head=head;self.layout=layout;self.copied=set();self.stamped=False;self.sealed=False
 def layer(self,l):
  if not self.lb<=l<self.le or l in self.copied:raise ValueError('duplicate/foreign layer')
  self.copied.add(l)
 def logits(self):
  if not self.head or 48 in self.copied:raise ValueError('foreign/duplicate head')
  self.copied.add(48)
 def stamp(self):
  if self.stamped or self.copied!=set(range(self.lb,self.le))|({48} if self.head else set()):raise ValueError('stamp before exact stage producers')
  self.stamped=True
 def seal(self,layout):
  if layout!=self.layout or not self.stamped:raise ValueError('graph seal/stamp mismatch')
  self.sealed=True

def route(source,lb,le,observe=True):
 """Read actual return-path hook presence; arithmetic/device calls unexecuted."""
 layout=(1,1,0,0);r=Roster(lb,le,le==48,layout)
 for l in range(lb,le):r.layer(l)
 early=source[source.index('    if (le_ < g.n_layers) {'):source.index('    // ---- the head, T columns')]
 if le<48:
  if observe and p.HOOK.strip() in early:r.stamp()
 else:
  r.logits()
  if observe and source.count(p.HOOK.strip())>=1:r.stamp()
 if observe:r.seal(layout)
 return r

class Controls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.old,cls.new=p.reconstruct()
 def test_actual35_stage0_failure_and_head_success(self):
  with self.assertRaisesRegex(ValueError,'seal/stamp'):route(self.old,0,32)
  self.assertTrue(route(self.old,32,48).sealed)
 def test_new37_both_stage_rosters_sealed(self):
  for lb,le in [(0,32),(32,48),(0,16),(16,48),(0,48)]:self.assertTrue(route(self.new,lb,le).sealed)
 def test_hook_after_all_three_handoff_copies_before_return(self):
  early=self.new[self.new.index('    if (le_ < g.n_layers) {'):self.new.index('    // ---- the head, T columns')]
  self.assertEqual(early.count('copy_from_mapped('),3);self.assertEqual(early.count(p.HOOK.strip()),1)
  self.assertLess(early.rindex('copy_from_mapped('),early.index(p.HOOK.strip()));self.assertLess(early.index(p.HOOK.strip()),early.index('return true;'))
 def test_exact_single_guarded_addition_no_other_source_change(self):
  self.assertEqual(self.new.replace(p.ANCHOR.replace('        return true;',p.HOOK+'        return true;'),p.ANCHOR),self.old)
  self.assertEqual(self.new.count(p.HOOK.strip()),2)
  self.assertEqual(self.new[self.new.index('    // ---- the head, T columns'):],self.old[self.old.index('    // ---- the head, T columns'):])
 def test_off_no_observer_stamp_or_seal(self):
  for lb,le in [(0,32),(32,48)]:
   r=route(self.new,lb,le,False);self.assertFalse(r.stamped);self.assertFalse(r.sealed)
 def test_missing_duplicate_foreign_producer_reject(self):
  r=Roster(0,32,False,(1,1,0,0));r.layer(0)
  with self.assertRaises(ValueError):r.stamp()
  with self.assertRaises(ValueError):r.layer(0)
  with self.assertRaises(ValueError):r.layer(32)
  with self.assertRaises(ValueError):r.logits()
 def test_wrong_layout_missing_head_duplicate_stamp_reject(self):
  r=Roster(32,48,True,(1,1,0,0))
  for l in range(32,48):r.layer(l)
  with self.assertRaises(ValueError):r.stamp()
  r.logits();r.stamp()
  with self.assertRaises(ValueError):r.stamp()
  with self.assertRaises(ValueError):r.seal((1,1,1,0))
 def test_real_observer_stage_queue_and_seal_guards_unchanged(self):
  h=(p.SDK/'sycl/include/strata/core/batch_fidelity_observer.hpp').read_text()
  self.assertIn('q.get_context()!=q_.get_context()||q.get_device()!=q_.get_device()',h)
  self.assertIn('i!=building_||i<0||!graphs_[i].stamped',h)
  self.assertIn('l>=lb_&&l<le_ || (l==LAYERS&&head_)',h)
  self.assertIn('batch_fidelity::snapshot>(*cs_,device_,int(lb_),int(le_),le_==g_->n_layers)',self.new)
 def test_plan_complete_closure_and_patch_identity(self):
  plan=p.build_plan();self.assertEqual((len(plan['expected_patched_source_sha256']),len(plan['added_header_payloads']),len(plan['patches']),len(plan['build_targets']),len(plan['runtime_python_sources'])),(63,27,36,8,6))
  self.assertEqual(plan['expected_patched_source_sha256'][p.V],hashlib.sha256(self.new.encode()).hexdigest())
  self.assertEqual(p.PATCH.read_bytes(),p.patch_bytes());self.assertFalse(plan['actual_paired_batch_observer37_qualified'])
 def test_optional_trace36_composition_preserves_every_other_byte(self):
  import prepare_host_critical_path36_v1 as h
  old,new=h.reconstruct();plan=p.combined_plan()
  self.assertEqual((len(plan['expected_patched_source_sha256']),len(plan['added_header_payloads']),len(plan['patches'])),(64,28,37))
  for name,data in new.items():
   expected=data.replace(p.ANCHOR,p.ANCHOR.replace('        return true;',p.HOOK+'        return true;')) if name==p.V else data
   self.assertEqual(plan['expected_patched_source_sha256'][name],hashlib.sha256(expected.encode()).hexdigest())
if __name__=='__main__':unittest.main()
