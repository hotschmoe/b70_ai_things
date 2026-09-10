"""CPU-only scale-receipt and positive-cache stage gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import fp8_cache_stage as stage


class StageTests(unittest.TestCase):
    def receipt(self):
        return dict(artifact_sha256='fixture',role='target',tp_rank=0,tp_size=1,
                    backend='triton',kv_dtype='fp8_e4m3',coverage=16,
                    rows=[{'query_scale_applied':False} for _ in range(16)])

    def test_receipt_requires_coverage_and_identity(self):
        receipt=self.receipt()
        self.assertEqual(stage.loaded_target('prefix B70_SGLANG_KV_SCALES_LOADED '+json.dumps(receipt),'fixture')['coverage'],16)
        receipt['coverage']=15
        with self.assertRaises(AssertionError):stage.loaded_target('B70_SGLANG_KV_SCALES_LOADED '+json.dumps(receipt),'fixture')

    def test_metrics_ignore_other_models_and_storage(self):
        rows=['sglang:cached_tokens_total{model_name="hotschmoe-dd",cache_source="device"} 4',
              'sglang:cached_tokens_total{model_name="other",cache_source="device"} 200',
              'sglang:cached_tokens_total{model_name="hotschmoe-dd",cache_source="storage"} 300']
        self.assertEqual(stage.cached_tokens('\n'.join(rows)),4)

    def exercise(self,tiny_rc,cache_after):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'viability').mkdir();(root/'tools').mkdir()
            (root/'server.log').write_text('B70_SGLANG_KV_SCALES_LOADED '+json.dumps(self.receipt()))
            (root/'viability/OUTCOME.json').write_text(json.dumps(dict(passed=True,requests=26)))
            (root/'tools/summary.json').write_text(json.dumps(dict(passed=True,checks=32,attempts=32,bang_attempts=0)))
            metric='sglang:cached_tokens_total{model_name="hotschmoe-dd",cache_source="device"} '
            (root/'tools/metrics-before.txt').write_text(metric+'4')
            (root/'tools/metrics-after.txt').write_text(metric+str(cache_after))
            plan=root/'stage.json';plan.write_text(json.dumps(dict(server_output=str(root),artifact_sha256='fixture',stages=[dict(name=name,command=[name],timeout=1500) for name in ['tiny-text','concurrent-cache']])))
            with patch('sys.argv',['stage',str(plan)]),patch.object(stage.subprocess,'run',side_effect=[NS(returncode=tiny_rc),NS(returncode=0)]) as run:
                rc=stage.main();calls=run.call_count
            return rc,calls,json.loads((root/'fp8-cache-stage-outcome.json').read_text())

    def test_tiny_failure_stops_cache_probe(self):
        rc,calls,result=self.exercise(1,12)
        self.assertEqual((rc,calls),(1,1));self.assertFalse(result['passed'])

    def test_positive_delta_required_after_all_checks(self):
        self.assertEqual(self.exercise(0,12)[:2],(0,2))
        rc,calls,result=self.exercise(0,4)
        self.assertEqual((rc,calls),(1,2));self.assertEqual(result['cached_token_delta'],0)


if __name__=='__main__':unittest.main()
