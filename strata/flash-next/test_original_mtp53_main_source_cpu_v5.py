import unittest,struct
from pathlib import Path
import prepare_original_mtp53_main_compile_v5 as m
class BytesControl(unittest.TestCase):
 def test_exact_comparator_types_and_all_fields(self):
  s=(m.HERE/'original-mtp53/loaded_main_adapter_v4.hpp').read_text()
  self.assertIn('same_bytes(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b)',s)
  self.assertIn('a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size())==0)',s)
  for key in ('gdn','ple','tails','dead','block_pos'):self.assertIn('!same_bytes(restored.'+key+',checkpoint.'+key+')',s)
  p=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/include/strata/core/conversation_cache.hpp')
  self.assertIn('std::vector<uint8_t> gdn, ple, tails, dead, block_pos;',p.read_text())
  self.assertNotEqual(struct.pack('<f',0.0),struct.pack('<f',-0.0))
 def test_delta_is_only_storage_comparator(self):
  old=(m.HERE/'original-mtp53/loaded_main_adapter_v3.hpp').read_text();new=(m.HERE/'original-mtp53/loaded_main_adapter_v4.hpp').read_text()
  line=next(r for r in new.splitlines(keepends=True)if 'static bool same_bytes' in r);new=new.replace(line,'')
  for key in ('gdn','ple','tails','dead','block_pos'):new=new.replace('!same_bytes(restored.'+key+',checkpoint.'+key+')','restored.'+key+'!=checkpoint.'+key)
  self.assertEqual(new,old)
