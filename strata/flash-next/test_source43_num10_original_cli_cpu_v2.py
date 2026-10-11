import ast,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_source43_num10_original_reference_v2 as c
import test_source43_num10_original_cli_cpu_v1 as old
class Controls(old.Controls):
 def setUp(self):self.p=patch.object(old,'c',c);self.p.start()
 def tearDown(self):self.p.stop()
class RoutingControls(unittest.TestCase):
 def test_exact_CLI_changes_callback_reference_and_plan_only(self):
  before=Path(c.__file__).with_name('run_source43_num10_original_reference_v1.py').read_text();expected=before.replace('import source43_num10_original_reference_v3 as reference','import source43_num10_original_reference_v4 as reference').replace('source43-num10-original-cli-source-plan-v1.json','source43-num10-original-cli-source-plan-v2.json').replace('from validate_source43_num10_zero_v3 import captured_binding','from validate_source43_num10_zero_v4 import captured_binding');self.assertEqual(Path(c.__file__).read_text(),expected)
 def test_reference_math_closing_source_exact_old3(self):
  before=Path(c.reference.__file__).with_name('source43_num10_original_reference_v3.py').read_text();self.assertEqual(Path(c.reference.__file__).read_text(),before.replace('source43-num10-original-reference-source-plan-v3.json','source43-num10-original-reference-source-plan-v4.json'))
 def test_actual_callback_import_calls_corrected_reader(self):
  import validate_source43_num10_zero_v4 as reader
  descriptor={'actual_callback_fixture':True,'requests':[]};calls=[]
  def enclosed(root,model_factory,capture_admit,closing,conditional):
   self.assertEqual(capture_admit(root),descriptor);calls.append('enclosed_after_callback');raise ValueError('controlled boundary after callback')
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor=root/'actor';actor.mkdir();(actor/'report.json').write_text('{}');identity=root/'identity';identity.write_text('{}');report_sha=hashlib.sha256((actor/'report.json').read_bytes()).hexdigest()
   with patch.object(c,'source_binding',return_value='SOURCE'),patch.object(c.reference,'enclosed',side_effect=enclosed),patch.object(reader,'captured_binding',side_effect=lambda r:(calls.append('corrected_reader4')or descriptor)),patch.object(c.reference,'load_targets',return_value=[]):
    with self.assertRaises(ValueError):c.execute(actor,report_sha,identity,hashlib.sha256(identity.read_bytes()).hexdigest(),root/'new')
   self.assertEqual(calls,['corrected_reader4','enclosed_after_callback'])
if __name__=='__main__':unittest.main()
