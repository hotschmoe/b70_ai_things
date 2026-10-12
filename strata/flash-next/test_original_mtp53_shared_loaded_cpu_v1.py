import unittest
from pathlib import Path
H=Path(__file__).resolve().parent/'original-mtp53'
class Private(unittest.TestCase):
 def test_bounded_full_state_and_no_cache_import(self):
  source=(H/'shared_weights_sessions_v3.hpp').read_text()
  for field in ['b.key','b.value','b.tail','b.dead','b.pooled','b.block_pos','committed_h','S.logits']:self.assertIn(field,source[source.index('OriginalDraftPrivateSnapshot53 diagnostic_snapshot'):])
  self.assertIn('q.wait_and_throw();',source[source.index('OriginalDraftPrivateSnapshot53 diagnostic_snapshot'):])
  self.assertNotIn('conversation_kv',source)
 def test_one_pack_and_actual_full_comparisons(self):
  source=(H/'shared_sessions_loaded_fixture_v1.cpp').read_text();self.assertEqual(source.count('std::make_shared<OriginalDraftWeights53>'),1)
  self.assertIn('std::array<std::unique_ptr<OriginalDraftSession53>,4>',source)
  self.assertIn('prefix_heads[lane][step]',source)
  for field in ['a.committed,b.committed','a.working,b.working','a.hidden,b.hidden','a.logits,b.logits']:self.assertIn(field,source)
  self.assertIn('slots[i]->drain(who[i]);slots[i].reset()',source)
  self.assertIn('lifetime.expired()',source)
