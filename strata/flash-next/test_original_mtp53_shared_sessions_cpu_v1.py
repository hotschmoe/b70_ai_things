"""Source-equivalence/lifetime controls only, no native or model execution."""
import unittest
from pathlib import Path
H=Path(__file__).resolve().parent/'original-mtp53'
class SharedSplit(unittest.TestCase):
 def setUp(self):self.old=(H/'loaded_main_adapter_pair_v3.hpp').read_text();self.new=(H/'shared_weights_sessions_v1.hpp').read_text()
 def test_exact_original_weights_and_private_allocations(self):
  a='    auto& source=*source_keep_;';b='    auto fl=[&]'
  self.assertIn(self.old[self.old.index(a):self.old.index(b)],self.new)
  alloc=self.old[self.old.index(b):self.old.index('\n\n }')].replace('q.wait_and_throw();source_keep_.reset();','q.wait_and_throw();')
  self.assertIn(alloc,self.new)
  pack=self.new.split('class OriginalDraftSession53',1)[0]
  self.assertNotIn('DraftBank committed=bank()',pack)
  self.assertEqual(self.new.count('source.load(path,error)'),1)
 def test_mapping_and_shared_owner_retirement_order(self):
  pack=self.new.split('class OriginalDraftSession53',1)[0]
  self.assertLess(pack.index('std::shared_ptr<sycl::queue> queue_keep_'),pack.index('std::unique_ptr<strata::core::original_mtp53::TensorViews> source_keep_'))
  self.assertLess(pack.index('source_keep_;'),pack.index('OwnedRuntime53 owner;'))
  self.assertNotIn('source_keep_.reset()',self.new)
  session=self.new.split('class OriginalDraftSession53',1)[1]
  self.assertLess(session.index('weights_keep_;'),session.index('OwnedRuntime53 owner;'))
  self.assertIn('~OriginalDraftSession53(){const strata::core::OnDevice on(1);owner.retire();}',session)
  owner=(H/'owned_runtime_pair_v1.hpp').read_text();self.assertLess(owner.index('q.wait_and_throw()'),owner.index('sycl::free(p,context)'))
  self.assertIn('std::_Exit(86)',owner)
 def test_slot_identity_and_private_target_staging(self):
  self.assertIn('who.rid!=slot_.rid || who.generation!=slot_.generation',self.new)
  self.assertIn('last.get_context()!=queue().get_context()',self.new)
  self.assertIn('canonical_path!=resolved',self.new)
  self.assertIn('publisher_sha256!="8087dbb',self.new)
  self.assertLess(self.new.index('q.memcpy(incoming,actual_last_hidden'),self.new.index('draft->record(token,incoming'))
  self.assertIn('catch(...){failed=true;throw;}',self.new)
  self.assertNotIn('conversation_snapshot',self.new)
  self.assertNotIn('main.run(',self.new)
if __name__=='__main__':unittest.main()
