import unittest
import run_original_mtp53_greedy64_v2 as r
class DockerAbsence(unittest.TestCase):
 def test_actual_lowercase_and_original_name(self):
  name='b70-mtp53-greedy-694770-0'
  for text in ('error: no such object: '+name+'\n','Error: No such object: '+name,'Error response from daemon: No such container: '+name):self.assertEqual(r.absence_classification(text.encode(),name),'original_absent')
  for text in ('error: no such object: foreign','daemon unavailable','error: no such object: '+name+'; unexpected','error: no such object: '+name+'\nextra'):self.assertEqual(r.absence_classification(text.encode(),name),'unknown')
 def test_bounded_unknown_classification(self):
  self.assertEqual(r.absence_classification(b'x'*4097,'name'),'unknown')
  self.assertEqual(r.absence_classification(b'\xff','name'),'unknown')
