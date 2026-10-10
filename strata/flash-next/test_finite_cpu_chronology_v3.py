"""Actual-shaped tiny receipt joins; no source/model/runtime observations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import finite_cpu_chronology_v3 as gate


class Controls(unittest.TestCase):
    def fixture(self, root):
        def put(path, row):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row))
        report = {'started_epoch': 1., 'finished_epoch': 110., 'cases': []}
        put(root / 'pre-hash-source-pages.json', {'passed': True, 'epoch': 2.})
        put(root / 'pre-full-four.json', {'passed': True, 'started_epoch': 3., 'finished_epoch': 4.})
        put(root / 'pre-inference-source-pages.json', {'passed': True, 'epoch': 5.})
        for i in range(16):
            case, repeat = divmod(i, 2)
            row = dict(case=case, repeat=repeat, passed=True, container_removed=True,
                       container_terminal_observed=True, errors=[], memory_errors=[],
                       started_epoch=7. + 4 * i, finished_epoch=8. + 4 * i)
            put(root / ('case%d-repeat%d' % (case, repeat)) / 'case-receipt.json', row)
            report['cases'].append(row)
            put(root / ('case%drepeat%d-pre-source-pages.json' % (case, repeat)), {'passed': True, 'epoch': 6. + 4 * i})
            put(root / ('case%drepeat%d-post-source-pages.json' % (case, repeat)), {'passed': True, 'epoch': 9. + 4 * i})
        report['last_case_terminal_epoch'] = 68.
        put(root / 'post-terminal-pre-hash-source-pages.json', {'passed': True, 'epoch': 80.})
        put(root / 'post-terminal-full-four.json', {'passed': True, 'started_epoch': 81., 'finished_epoch': 90.})
        put(root / 'post-terminal-post-hash-source-pages.json', {'passed': True, 'epoch': 91.})
        return report

    def test_complete_original16_boundary_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            self.assertEqual(gate.admit(root, report)['derived_last_case_terminal_epoch'], 68.)

    def test_forged_parent_boundary_and_posthash_before_actual_last_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            bad = copy.deepcopy(report)
            bad['last_case_terminal_epoch'] = 8.
            self.assertRaises(ValueError, gate.admit, root, bad)
            (root / 'post-terminal-full-four.json').write_text(json.dumps({'passed': True, 'started_epoch': 60., 'finished_epoch': 90.}))
            self.assertRaises(ValueError, gate.admit, root, report)

    def test_original_receipt_type_roster_and_page_scope_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            path = root / 'case0-repeat0/case-receipt.json'
            bad = copy.deepcopy(report['cases'][0])
            bad['passed'] = 1
            path.write_text(json.dumps(bad))
            self.assertRaises(ValueError, gate.admit, root, report)
            report = self.fixture(root)
            (root / 'case7repeat1-post-source-pages.json').write_text(json.dumps({'passed': True, 'epoch': 67.}))
            self.assertRaises(ValueError, gate.admit, root, report)
            report['cases'][0]['finished_epoch'] = float('nan')
            self.assertRaises(ValueError, gate.admit, root, report)
