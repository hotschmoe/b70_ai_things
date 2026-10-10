"""Synthetic saved-receipt controls; no native or original model execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import original_head_distribution_v2 as q

class SavedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        heads = {}
        for name in ('head', 'native_head'):
            path = self.root/(name+'.f32')
            path.write_bytes(np.linspace(-1, 1, 248320, dtype='<f4').tobytes())
            heads[name] = {'path': str(path), 'sha256': q.digest(path), 'bytes': 993280}
        self.report = {
            'status': 'exploratory complete; NO numerical qualification', 'errors': [],
            'full_model_math_qualified': False, 'dependency_sha256': {'CPU': 'synthetic'},
            'native_observation_binding': {'prefix': 2,
                'native_inputs_or_routes_used_for_own_computation': False,
                'native_vectors': {'head': heads['native_head']}},
            'post_original_identity': {'complete_four_publisher_hashes_verified': True,
                                       'current_stat_verified': True},
            'post_dispatch_binding': {'CPU': 'synthetic'},
            'source_dispatch_binding': {'CPU': 'synthetic'},
            'exploration': {'arrays': heads}}
        self.path = self.root/'synthetic-report.json'

    def analyze(self, report=None):
        self.path.write_text(json.dumps(report or self.report))
        with patch.object(q, 'dependency_binding', return_value={'CPU': 'synthetic'}):
            return q.analyze(self.path, q.digest(self.path))

    def test_bound_complete_saved_heads_and_no_quality_claim(self):
        result = self.analyze()
        self.assertTrue(result['argmax_equal'])
        self.assertEqual(result['total_variation'], 0)
        self.assertFalse(result['quality_qualified'])
        self.assertFalse(result['full_model_math_qualified'])

    def test_incomplete_or_borrowed_or_stale_source_refused(self):
        for field, value in [('status', 'computed; final full4 source proof pending'),
                             ('errors', ['incomplete']), ('dependency_sha256', {}),
                             ('post_dispatch_binding', {'CPU': 'changed'})]:
            report = copy.deepcopy(self.report)
            report[field] = value
            with self.assertRaises(ValueError): self.analyze(report)
        report = copy.deepcopy(self.report)
        report['native_observation_binding']['native_inputs_or_routes_used_for_own_computation'] = True
        with self.assertRaises(ValueError): self.analyze(report)

    def test_copied_head_must_match_admitted_native_capture(self):
        report = copy.deepcopy(self.report)
        report['native_observation_binding']['native_vectors']['head'] = {'sha256': 'f'*64}
        with self.assertRaises(ValueError): self.analyze(report)

    def test_current_saved_raw_mutation_refused(self):
        path = Path(self.report['exploration']['arrays']['head']['path'])
        path.write_bytes(bytes(993280))
        with self.assertRaises(ValueError): self.analyze()

if __name__ == '__main__':
    unittest.main()
