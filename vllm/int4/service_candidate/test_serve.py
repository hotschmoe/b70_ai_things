import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import serve


class Gates(unittest.TestCase):
    def test_unqualified_refused_before_device_work(self):
        with self.assertRaisesRegex(RuntimeError, 'not qualified'):
            serve.validate_qualification({}, {'qualified': False})

    def test_missing_workload_evidence_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'plan.json'
            p.write_text(json.dumps({'out': d, 'jobs': []}))
            with self.assertRaisesRegex(RuntimeError, 'workloads incomplete'):
                serve.evidence(p)

    def test_changed_input_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'source'; p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                serve.verify_inputs({'sha256': {str(p): '0' * 64}})

    def test_exact_candidate_command(self):
        inputs = serve.read(serve.INPUTS)
        plan = serve.verify_inputs(inputs)
        cmd = serve.service_command(plan, Path('/temporary/server'))
        original = plan['server'][:]
        self.assertNotIn('--health-p2p-check', cmd)
        self.assertEqual(serve.value(cmd, '--p2p'), '0')
        self.assertEqual(serve.value(cmd, '--served-model'), 'hotschmoe-dd')
        self.assertEqual(serve.value(cmd, '--port'), '18124')
        normalized = cmd[:]; normalized.remove('--leased')
        i = normalized.index('--port'); del normalized[i:i+2]
        for flag in ['--out', '--name']:
            normalized[normalized.index(flag)+1] = original[original.index(flag)+1]
        self.assertEqual(normalized, original)

    def test_wrong_image_refused(self):
        inputs = serve.read(serve.INPUTS)
        plan = serve.read(inputs['plan200k']); plan['server'][plan['server'].index('--image')+1] = 'wrong'
        realread = serve.read
        with patch.object(serve, 'read', side_effect=lambda p: plan if str(p) == inputs['plan200k'] else realread(p)):
            with self.assertRaisesRegex(RuntimeError, 'wrong candidate flag'):
                serve.verify_inputs(inputs)

    def test_bounded_startup_scope_and_namespaces(self):
        inputs = serve.read(serve.INPUTS)
        jobs = serve.startup_jobs(inputs, Path('/fresh/server'), 'fresh-run')
        self.assertEqual(len(jobs), 5)
        self.assertEqual(jobs[0]['name'], '02-tiny24')
        rounds = [j for j in jobs if '--temperature' in j['command']]
        self.assertEqual([serve.value(j['command'], '--temperature') for j in rounds], ['0', '0.7'])
        self.assertEqual(len({serve.value(j['command'], '--cache-namespace') for j in rounds}), 2)
        for j in rounds:
            self.assertEqual(serve.value(j['command'], '--records'), '360')
            self.assertEqual(serve.value(j['command'], '--base'), 'http://127.0.0.1:18124')
        self.assertTrue(all('/fresh/server' in str(j['command']) for j in jobs))
        self.assertTrue(all('long_context_qualification' not in str(j['command']) for j in jobs))
        clean = serve.read(inputs['plan100k'])
        for j in jobs:
            original = next(x for x in clean['jobs'] if x['name'] == j['name'])
            self.assertEqual(j['command'][1], original['command'][1])

    def test_unowned_frontdoor_refused(self):
        with patch.object(serve.Path, 'iterdir', return_value=iter([])), patch.object(serve.Path, 'read_text', return_value='header\n 0: 00000000:46A0 00000000:0000 0A 0:0 00:0 0 0 0 12345\n'):
            self.assertFalse(serve.owns_frontdoor(123))

    def test_pointer_escape_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid service pointer'):
            serve.stop_result(Path('/tmp/unrelated'))


if __name__ == '__main__':
    unittest.main()
