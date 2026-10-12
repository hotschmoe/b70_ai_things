import unittest
from pathlib import Path
import prepare_original_mtp53_main_compile_v4 as m
class SnapshotControl(unittest.TestCase):
 def test_saved_state_surrounds_actual_control_and_rejoins(self):
  s=(m.HERE/'original-mtp53/loaded_main_adapter_v3.hpp').read_text()
  a=s.index('conversation_checkpoint_save(checkpoint');b=s.index('const bool control_ok=main.run');c=s.index('conversation_checkpoint_restore(checkpoint');d=s.index('conversation_kv_restore(kv');e=s.index('conversation_kv_verify(kv');f=s.index('draft->verify_and_catchup(main')
  self.assertTrue(a<b<c<d<e<f);self.assertIn('conversation_kv_save(kv',s[a:b])
  self.assertIn('if(!control_ok)throw std::runtime_error(control_error)',s[e:f])
  for field in ('gdn','ple','tails','dead','block_pos'):self.assertIn('restored.'+field+'!=checkpoint.'+field,s)
  self.assertIn('ONE_TOKEN_COMMIT=0',s);self.assertIn('target_history.push_back(input);if(accepted)target_history.push_back(guess)',s)
 def test_real_sdk_snapshot_signatures_and_carve(self):
  p=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/include/strata/core/conversation_snapshot.hpp');s=p.read_text()
  for name in ('conversation_checkpoint_save','conversation_checkpoint_restore','conversation_kv_save','conversation_kv_restore','conversation_kv_verify'):self.assertIn(name,s)
  self.assertIn('const SessionState& session',s);self.assertIn('uint64_t& fingerprint',s)
  preparer=Path(m.__file__).read_text();self.assertIn('exercise(ver,ss,g,win_pool_fn,win_pool_user)',preparer)
