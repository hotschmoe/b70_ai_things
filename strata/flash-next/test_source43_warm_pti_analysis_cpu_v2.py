import ast,unittest
from pathlib import Path
from unittest.mock import patch
import analyze_source43_warm_pti_v2 as a
import test_source43_pti_analysis_cpu_v1 as old
class Controls(old.Controls):
    def setUp(self):self.p=patch.object(old,'a',a);self.p.start()
    def tearDown(self):self.p.stop()
class NewControls(unittest.TestCase):
    def test_exact_warm_source_and_wrong_generation(self):
        expected=a.host.bound(a.warm_actor.PLAN)['sha256'];report={'source_plan_sha256':expected};admit={'source_plan_sha256':expected,'source_generation':43,'materialized_mode':0};self.assertEqual(a.generation_gate(report,admit),expected)
        for key,value in(('source_generation',41),('materialized_mode',False),('source_plan_sha256','cold')):
            with self.assertRaises(ValueError):a.generation_gate(report,{**admit,key:value})
    def test_selected_only_metric_dispatch(self):
        row={'ordinal':2,'raw':{'ids':[1]*54,'output_ids':[2]*4,'done':'DONE 4 54 1 2 length'}}
        with patch.object(a.host,'metrics',return_value={'request_completion_ns':123})as metrics:
            self.assertEqual(a.selected_host_metric(row,['actual'])['request_completion_ns'],123);metrics.assert_called_once_with(['actual'])
        with patch.object(a.host,'metrics')as metrics:
            with self.assertRaises(ValueError):a.selected_host_metric({**row,'ordinal':1},[])
            metrics.assert_not_called()
    def test_raw_interval_math_identical(self):
        def node(file):return next(n for n in ast.parse(Path(file).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='raw_device_metrics')
        self.assertEqual(ast.dump(node(a.__file__)),ast.dump(node(Path(a.__file__).with_name('analyze_source43_pti_latency_v2.py'))))
    def test_scope_and_no_standalone_model_admission(self):
        text=Path(a.__file__).read_text();self.assertIn('warm_public.control_binding',text);self.assertIn("'device_interval_scope'",text);self.assertIn('including possible teardown',text);self.assertNotIn('warm_public.finalized_binding(',text);self.assertIn("'bandwidth_qualified':False",text)
if __name__=='__main__':unittest.main()
