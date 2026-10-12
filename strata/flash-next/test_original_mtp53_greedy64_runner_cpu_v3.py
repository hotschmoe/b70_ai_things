import unittest
from pathlib import Path
import run_original_mtp53_greedy64_v3 as r
class NativeRoute(unittest.TestCase):
 def test_exact_two_source_pin_changes(self):
  b=r.HERE;s=(b/'run_original_mtp53_greedy64_v3.py').read_text();old=(b/'run_original_mtp53_greedy64_v2.py').read_text()
  s=s.replace('original-publisher-mtp53-greedy64-runner-proposal-v3.json','original-publisher-mtp53-greedy64-runner-proposal-v2.json').replace('original-publisher-mtp53-greedy64-proposal-v2.json','original-publisher-mtp53-greedy64-proposal-v1.json');self.assertEqual(s,old)
  self.assertEqual(r.absence_classification(b'error: no such object: owned\n','owned'),'original_absent');self.assertEqual(r.absence_classification(b'error: no such object: other\n','owned'),'unknown')
