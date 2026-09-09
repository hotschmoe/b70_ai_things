import json
from pathlib import Path
import tempfile
import unittest
from continue_initial import decision


class ContinuationGate(unittest.TestCase):
    def test_only_timeout_can_retry_and_health_dominates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'exit.rc').write_text('0\n')
            (root / 'arm-results.json').write_text(json.dumps({'01-quality': 0, '01b-guides': 124}))
            self.assertEqual(decision(root), 'retry_timeout')
            (root / 'arm-results.json').write_text(json.dumps({'01-quality': 0, '01b-guides': 1}))
            with self.assertRaises(RuntimeError):
                decision(root)
            (root / 'WORKLOADS_PASSED').touch()
            self.assertEqual(decision(root), 'freeze')
            (root / 'exit.rc').write_text('1\n')
            with self.assertRaises(RuntimeError):
                decision(root)
