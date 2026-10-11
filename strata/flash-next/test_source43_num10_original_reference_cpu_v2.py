import unittest
from unittest.mock import patch
import source43_num10_original_reference_v2 as q
import test_source43_num10_original_reference_cpu_v1 as old
class Controls(old.Controls):
 def setUp(self):self.p=patch.object(old,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class RawControls(old.RawControls):
 def setUp(self):self.p=patch.object(old,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class ClosingControls(old.ClosingControls):
 def setUp(self):self.p=patch.object(old,'q',q);self.p.start()
 def tearDown(self):self.p.stop()
class NewControls(unittest.TestCase):
 def test_arithmetic_and_target_functions_byte_semantics_unchanged(self):
  import ast
  from pathlib import Path
  left=ast.parse(Path(old.q.__file__).read_bytes());right=ast.parse(Path(q.__file__).read_bytes());a={n.name:ast.dump(n)for n in left.body if isinstance(n,ast.FunctionDef)and n.name!='enclosed'};b={n.name:ast.dump(n)for n in right.body if isinstance(n,ast.FunctionDef)and n.name!='enclosed'};self.assertEqual(a,b)
 def test_first_admission_failure_still_attempts_both_closings(self):
  calls=[]
  def capture(root):calls.append('capture');raise ValueError('first admission failed')
  def scan(boundary):calls.append('scan');raise ValueError('closing scan failed')
  with patch.object(q,'source_binding',return_value='source'):
   with self.assertRaisesRegex(ValueError,'first admission failed'):q.enclosed('root',lambda:None,capture,scan)
  self.assertEqual(calls,['capture','capture','scan'])
 def test_scan_after_failed_closing_capture_actual_read_boundary(self):
  clock=[100.];calls=[]
  def capture(root):
   calls.append('capture');clock[0]+=10
   if len(calls)==2:raise ValueError('failed after model read')
   return {'requests':[]}
  def scan(boundary):calls.append('scan');self.assertEqual(boundary,120.);raise ValueError('scan failed')
  with patch.object(q,'source_binding',return_value='source'),patch.object(q.time,'time',side_effect=lambda:clock[0]),patch.object(q,'load_targets',side_effect=ValueError('raw failed')):
   with self.assertRaisesRegex(ValueError,'failed after model read'):q.enclosed('root',lambda:None,capture,scan)
  self.assertEqual(calls,['capture','capture','scan'])
 def test_stale_scan_started_after_compute_before_closing_capture_refused(self):
  import original_first_gdn_layer_v1 as equations
  class Model:pass
  clock=[100.];captures=[]
  def capture(root):captures.append(1);clock[0]+=10;return {'requests':[]}
  def scan(boundary):
   self.assertEqual(boundary,120.)
   return {'rows':[],'started':115.,'finished':121.}
  with patch.object(q,'source_binding',return_value='source'),patch.object(q.time,'time',side_effect=lambda:clock[0]),patch.object(q,'load_targets',return_value=[]),patch.object(q,'evaluate',return_value={'computation_terminal_epoch':111.}),patch.object(equations,'FirstGdnLayer',Model):
   with self.assertRaisesRegex(ValueError,'Fresh post-closing-capture publisher boundary'):q.enclosed('root',Model,capture,scan)
  self.assertEqual(len(captures),2)
if __name__=='__main__':unittest.main()
