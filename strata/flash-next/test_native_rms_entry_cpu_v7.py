"""Source byte identity and nested-main-macro regression; no compilation."""
import unittest
from pathlib import Path
import native_rms_entry_derivation_v7 as d
import qualify_native_rms_rsqrt37_v7 as q
class Tests(unittest.TestCase):
 def raws(self):
  h=Path(__file__).parent;return [(h/n).read_bytes() for n in ['native_rms_rsqrt37_gpu_v1.cpp','native_rms_rsqrt37_callable_v7.cpp','native_rms_rsqrt37_owned_entry_v1.cpp','native_rms_rsqrt37_owned_entry_v7.cpp']]
 def test_exact_function_and_include_derivation(self):self.assertFalse(d.admit(Path(__file__).parent)['leaf_math_changed'])
 def test_any_additional_leaf_or_wrapper_change_rejected(self):
  for index in (1,3):
   raw=self.raws();raw[index]+=b'\n'
   with self.assertRaises(ValueError):d.prove(*raw)
 def test_missing_or_duplicated_function_identifier_rejected(self):
  raw=self.raws()
  for changed in (raw[1].replace(d.CALLABLE,d.FUNCTION),raw[1]+d.CALLABLE):
   with self.assertRaises(ValueError):d.prove(raw[0],changed,raw[2],raw[3])
 def test_original_inner_undef_macro_and_new_direct_callable(self):
  proposal,callable_source,wrapper,entry=self.raws();self.assertLess(proposal.index(b'#undef main'),proposal.index(d.FUNCTION));self.assertIn(b'#define main rms37_frozen_proposal_main',wrapper);self.assertNotIn(b'#define main rms37_frozen_proposal_main',entry);self.assertEqual(callable_source.count(d.CALLABLE),1);self.assertNotIn(d.FUNCTION,callable_source);self.assertEqual(entry.count(d.FUNCTION),1)
 def test_both_actual_and_expected_compile_use_new_entry(self):
  s=Path(q.__file__).read_text();self.assertEqual(s.count("'/leaf/native_rms_rsqrt37_owned_entry_v7.cpp'"),2);self.assertNotIn('/leaf/native_rms_rsqrt37_owned_entry_v1.cpp',s);self.assertIn('derivation=entry_admit(HERE)',s)
 def test_V6_frozen_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-owned-source-plan-v6.json')),'d87a7512e153a8426471f208bfc4ddf1d0f4bc5ac77662048b624732255a86cf')
if __name__=='__main__':unittest.main()
