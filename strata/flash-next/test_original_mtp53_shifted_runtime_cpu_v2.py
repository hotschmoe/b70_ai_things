import unittest
import run_original_mtp53_shifted_v2 as r
class Binds(unittest.TestCase):
 def test_real_reordered_mounts(self):
  a=['/build/output:/harness:ro','/model/original.gguf:/original.gguf:ro'];self.assertTrue(r.exact_binds(a[::-1],a))
 def test_duplicate_or_writable(self):
  a=['/a:/harness:ro','/b:/original.gguf:ro'];self.assertFalse(r.exact_binds([a[0],a[0]],a));self.assertFalse(r.exact_binds([a[0],'/b:/original.gguf:rw'],a))
 def test_types(self):self.assertFalse(r.exact_binds(('a',),['a']));self.assertFalse(r.exact_binds([1],['1']))
