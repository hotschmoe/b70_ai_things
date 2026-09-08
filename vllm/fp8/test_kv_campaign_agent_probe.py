"""CPU regression: one failed session must not erase other response evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import kv_campaign_agent_probe as probe


class EvidenceTest(unittest.TestCase):
    def exercise(self, failure):
        def request(base, path, payload=None, **kwargs):
            if path == '/v1/models':
                return json.dumps({'data': [{'id': 'test-model'}]})
            if path == '/metrics':
                return 'test_metric 0\n'
            index = int(payload['cache_salt'].rsplit('-', 1)[1])
            if index == 2 and failure == 'timeout':
                raise TimeoutError('injected deadline')
            if payload['tool_choice'] == 'none':
                count = json.loads(payload['messages'][-1]['content'])['count']
                message = {'content': str(count)}
            else:
                sku = payload['messages'][-1]['content'].split('SKU ', 1)[1].rstrip('.')
                arguments = 'malformed' if index == 2 else json.dumps({'sku': sku})
                message = {'content': None, 'tool_calls': [{'id': 'test-call', 'type': 'function',
                           'function': {'name': 'lookup_stock', 'arguments': arguments}}]}
            return json.dumps({'choices': [{'message': message}]})
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'probe'
            with patch.object(probe, 'request', request), patch('sys.argv', [
                    'probe', '--model', 'test-model', '--out', str(out), '--records', '1']):
                self.assertEqual(probe.main(), 1)
            rows = json.loads((out / 'results.json').read_text())
            self.assertEqual(len(rows), 4)
            self.assertEqual([r['passed'] for r in rows], [True, True, False, True])
            self.assertEqual(len(rows[2]['rows']), 1)
            saved = [json.loads(line) for p in out.glob('session-*.jsonl') for line in p.read_text().splitlines()]
            self.assertEqual(len(saved), 25)
            self.assertEqual(sum(r['passed'] for r in saved), 24)
            if failure == 'timeout':
                self.assertIn('injected deadline', rows[2]['rows'][0]['error'])
            else:
                self.assertIn('malformed', json.dumps(rows[2]['rows'][0]['response']))

    def test_timeout_preserves_other_sessions(self):
        self.exercise('timeout')

    def test_bad_tool_arguments_preserve_raw_response(self):
        self.exercise('arguments')


if __name__ == '__main__':
    unittest.main()
