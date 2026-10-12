import argparse,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_current59_first_gdn_reference_v1 as driver
class FailureClose(unittest.TestCase):
 def test_initial_target_failure_always_fresh_close(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);identity=root/'identity.json';identity.write_text('{}\n',encoding='ascii');a=argparse.Namespace(output=root/'out',model_identity=identity,expected_identity_sha256=driver.sha(identity),native_root=root,producer_report_sha256='0'*64,public_binding=root/'public.json',public_binding_sha256='0'*64)
   def reject(*args):raise ValueError('original target digest refused')
   with patch.object(driver,'source_binding',return_value='a'*64),patch.object(driver,'targets',side_effect=reject),patch('run_source44_num38_original_reference_v2.closing_identity',return_value={'passed':True})as close,patch.object(driver,'BoundOriginalFull48Rows')as provider:
    with self.assertRaises(ValueError):driver.execute(a)
    self.assertEqual(close.call_count,1);provider.assert_not_called()
    report=driver.read_unique(a.output/'report.json');self.assertFalse(report['passed']);self.assertFalse(report['captured_inputs_used']);self.assertFalse(report['numeric_pass_assigned']);self.assertIn('original target digest refused',report['errors'][0])
