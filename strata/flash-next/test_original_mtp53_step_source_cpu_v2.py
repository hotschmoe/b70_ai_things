"""Source/position controls only, following current generate/MtpDrafter target flow."""
import json,hashlib,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_accept_reject_correction_actual_pair_positions(self):
  # Existing draft(T,outv,p,a) catches up cells p..p+a using NEXT picks + MAIN R rows.
  for base,guess,picks in [(10,23,[23,41]),(10,23,[31,99])]:
   accepted=picks[0]==guess;keep=2 if accepted else 1;cache_cells=base
   catchup=[(base+r,picks[r],'main_R_'+str(r))for r in range(keep)]
   for cell,token,h in catchup:self.assertEqual(cell,cache_cells);cache_cells+=1
   self.assertEqual(cache_cells,base+keep);self.assertEqual(picks[keep-1],41 if accepted else 31)
   if not accepted:self.assertEqual(catchup,[(10,31,'main_R_0')]);self.assertNotIn(guess,[x[1]for x in catchup])
  text=(HERE/'original-mtp53/draft_step_v2.hpp').read_text();self.assertIn('base_position!=state_.committed_cells()',text);self.assertIn('main.final_R_all()+size_t(row)*10240',text);self.assertIn('if(active_)settle_fixture(false)',text);self.assertIn('main_failed_=true',text);self.assertIn('main.run(2,inputs.data()',text);self.assertIn('main.commit(keep',text)
 def test_failed_drain_never_frees_or_destroys_pending_graphs(self):
  text=(HERE/'original-mtp53/owned_runtime_v1.hpp').read_text();self.assertLess(text.index('q.wait_and_throw()'),text.index('graph_deleters.rbegin()'));self.assertLess(text.index('graph_deleters.rbegin()'),text.index('sycl::free'));self.assertIn('std::_Exit(86)',text)
  for name in ['eh_hc_loaded_fixture_v2.cpp','primitive_fixture_v2.cpp','draft_step_loaded_fixture_v1.cpp']:
   source=(HERE/'original-mtp53'/name).read_text();self.assertIn('keep_graph',source);self.assertIn('.retire()',source)
 def test_actual_queue_pointer_contract(self):
  for name in ['typed_primitives_v2.hpp','eh_hc_front_v2.hpp','qsa_state_segment_v2.hpp','draft_step_v2.hpp']:
   source=(HERE/'original-mtp53'/name).read_text();self.assertNotIn('auto& q=strata::q_of(',source);self.assertNotIn('strata::q_of(stream_).memcpy',source);self.assertIn('auto& q=*strata::q_of(',source)
 def test_full_new_overlay_apply_reverse(self):
  spec=json.loads((HERE/'original-publisher-mtp53-step-proposal-v2.json').read_bytes());patch=HERE/'patches/original-publisher-mtp53-step-v2.patch'
  self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(),spec['patch_sha256'])
  with tempfile.TemporaryDirectory()as d:
   subprocess.run(['git','apply',str(patch)],cwd=d,check=True,capture_output=True);self.assertEqual({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(d).rglob('*')if p.is_file()},spec['added_files']);subprocess.run(['git','apply','--reverse',str(patch)],cwd=d,check=True,capture_output=True);self.assertFalse(any(p.is_file()for p in Path(d).rglob('*')))
if __name__=='__main__':unittest.main()
