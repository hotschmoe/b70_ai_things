import json
from pathlib import Path
import unittest
from unittest.mock import patch
import serve


class Trial(unittest.TestCase):
    def test_both_context_configs_and_primary_alias(self):
        for ctx in ['100k', '200k']:
            inputs = serve.read(serve.HERE / ('inputs-' + ctx + '.json'))
            plan = serve.verify_trial(inputs)
            cmd = serve.service_command(plan, Path('/new/server'))
            self.assertEqual(serve.value(cmd, '--served-model'), 'hotschmoe-dd')
            self.assertEqual(serve.value(cmd, '--p2p'), '0')
            self.assertEqual(serve.value(cmd, '--port'), '18124')
            self.assertNotIn('--health-p2p-check', cmd)
            self.assertFalse(inputs['production_qualified'])

    def test_no_full_qualification_claim_allowed(self):
        with self.assertRaisesRegex(RuntimeError, 'must not claim'):
            serve.verify_trial({'user_authorized_day_trial': True, 'production_qualified': True})

    def test_tiny_only_startup(self):
        inputs = serve.read(serve.HERE / 'inputs-200k.json')
        jobs = serve.tiny_job(inputs, Path('/new/server'))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['name'], '02-tiny24')
        self.assertEqual(serve.value(jobs[0]['command'], '--base-url'), 'http://127.0.0.1:18124')

    def test_missing_partial_evidence_refused(self):
        with patch.object(serve, 'read', return_value={}):
            with self.assertRaises((KeyError, RuntimeError)):
                serve.trial_prerequisite({'prerequisite': '/missing'})


if __name__ == '__main__':
    unittest.main()
