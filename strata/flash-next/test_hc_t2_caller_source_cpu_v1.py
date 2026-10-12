import unittest,tempfile,subprocess
from pathlib import Path
import prepare_hc_t2_caller_v1 as p
class DeadBank(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.files=p.construct()
 def test_actual_capacity_and_source_lifetime(self):
  s=(p.SDK/'sycl/src/core/verify.cpp').read_text()
  self.assertIn('parts_ = b.take<float>(T * K * N)',s)
  self.assertGreaterEqual(2*10*2560,2*4*2560)
  combine=s.index('native_moe_combine_multi(parts_');join=s.index('cs->ext_oneapi_submit_barrier({*ev_join_});',s.index('auto post ='))
  self.assertLess(join,combine);self.assertGreater(s.index('if (l + 1 < le_ && !pre(l + 1, grp))'),combine)
  new=self.files['sycl/src/core/verify.cpp']
  for gate in ('T==2 && G==1','!batch_rec_','!prefix30_prompt_','!fidelity_first_','!remote_opt_','next_==nullptr','ext_stream_==nullptr','K>=HC'):self.assertIn(gate,new)
 def test_no_recurrence_or_commit_delta_and_no_allocation(self):
  old=(p.SDK/'sycl/src/core/verify.cpp').read_text();new=self.files['sycl/src/core/verify.cpp']
  a=old.index('bool Verifier::capture_commit');self.assertIn(old[a:],new)
  for token in ('gdn_step_norm_multi(','ple_history_advance(','kv_append_step(','b.take<float>','wait_and_throw(','queues_wait_and_throw('):self.assertEqual(old.count(token),new.count(token))
  self.assertIn('a.gate=parts_;a.gate_floats=size_t(2*HC*N)',new)
 def test_full_apply_reverse_undo(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d)
   for name in p.TARGETS:
    f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((p.SDK/name).read_bytes())
   patch=p.HERE/'patches/hc-t2-caller-reuse-v1.patch';subprocess.run(['git','apply',str(patch)],cwd=root,check=True,capture_output=True)
   for name,want in self.files.items():self.assertEqual((root/name).read_text(),want)
   subprocess.run(['git','apply','--reverse',str(patch)],cwd=root,check=True,capture_output=True)
   for name in p.TARGETS:self.assertEqual((root/name).read_bytes(),(p.SDK/name).read_bytes())
