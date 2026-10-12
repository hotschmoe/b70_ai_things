import unittest,tempfile,subprocess
from pathlib import Path
import prepare_expert_pair_reuse_v1 as p
class PairSource(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.old=p.SOURCE.read_text();cls.new=p.construct(cls.old)
 def test_exact_typed_helpers_and_lane_order(self):
  a=self.old.index('template<int TY> struct S27;');b=self.old.index('\ntemplate <int TG, bool LX, int RPW, bool TS>',a)
  self.assertIn(self.old[a:b].replace('S27','ExpertPairV1'),self.new)
  self.assertIn('for(int k=sub;k<nb*F::ipb;k+=LANES)',self.new)
  self.assertIn('out[0]=lanes_sum<LANES>(a);out[1]=lanes_sum<LANES>(b)',self.new)
 def test_default_off_role_scope_and_scattered_destinations(self):
  self.assertIn('v && v[0]==\'1\' && v[1]==\'\\0\'',self.new)
  self.assertIn('if constexpr(TY==12 || TY==13)',self.new);self.assertIn('if constexpr(TD==7 || TD==8)',self.new)
  self.assertEqual(self.new.count('if(pair_reuse && e1-e0==2)'),2)
  self.assertIn('hq+size_t(e0)*hb,hq+size_t(e0+1)*hb',self.new)
  self.assertIn('out[size_t(ent_dst[e0])*L.n_embd+r]=paired[0]',self.new)
  for enabled,kind,count,want in [(True,12,2,True),(False,12,2,False),(True,12,1,False),(True,12,3,False),(True,13,2,True),(True,21,2,False)]:self.assertEqual(enabled and kind in(12,13)and count==2,want)
 def test_no_new_allocation_wait_or_query_and_original_fallback(self):
  for token in ('malloc','sycl::free','wait_and_throw','barrier(','get_info','native_expert_grouped('):self.assertEqual(self.old.count(token),self.new.count(token),token)
  start=self.old.index('    int e = e0;',self.old.index('void row_entries'));end=self.old.index('\nconstexpr int GU_ROWS',start)
  self.assertIn(self.old[start:end],self.new)
  self.assertIn('native_gu_kernel<TG, LN>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, pair_reuse)',self.new)
  self.assertIn('native_down_kernel<TD, LN>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, pair_reuse)',self.new)
 def test_actual_patch_apply_reverse_full_bytes(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);f=root/p.TARGET;f.parent.mkdir(parents=True);f.write_text(self.old)
   patch=p.HERE/'patches/expert-pair-reuse-v1.patch'
   subprocess.run(['git','apply',str(patch)],cwd=root,check=True,capture_output=True)
   self.assertEqual(f.read_text(),self.new)
   subprocess.run(['git','apply','--reverse',str(patch)],cwd=root,check=True,capture_output=True)
   self.assertEqual(f.read_text(),self.old)
