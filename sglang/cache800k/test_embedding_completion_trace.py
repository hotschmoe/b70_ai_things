import contextlib
import io
import json
import types
import unittest
from unittest.mock import patch

import embedding_completion_trace as trace
from test_embedding_trace import Tensor


class CompletionTest(unittest.TestCase):
    def run_control(self, fail=None):
        tensor = Tensor()
        tensor.shape = (2048, 5120)
        tensor.numel = lambda: 2048 * 5120
        tensor.device = types.SimpleNamespace(type='xpu', index=1)
        output = io.StringIO()
        calls = []

        def synchronize(device):
            self.assertIs(device, tensor.device)
            calls.append('sync')
            if fail == len(calls):
                raise RuntimeError('sync failed')

        def collective(value):
            self.assertIs(value, tensor)
            calls.append('reduce')
            if fail == 'reduce':
                raise RuntimeError('collective failed')
            return value

        torch = types.SimpleNamespace(xpu=types.SimpleNamespace(synchronize=synchronize))
        with patch.dict('sys.modules', torch=torch), contextlib.redirect_stdout(output):
            if fail is None:
                self.assertIs(trace.reduce_call(collective, tensor, 1, 4274, 'tp'), tensor)
            else:
                with self.assertRaises(RuntimeError):
                    trace.reduce_call(collective, tensor, 1, 4274, 'tp')
        records = [json.loads(line.split(' ', 1)[1]) for line in output.getvalue().splitlines()]
        self.assertTrue(all(r['bytes'] == 20971520 and r['sequence'] == 4274 for r in records))
        return calls, records

    def test_order_and_completion_scope(self):
        calls, records = self.run_control()
        self.assertEqual(calls, ['sync', 'reduce', 'sync'])
        self.assertEqual([r['stage'] for r in records], [
            'pre_collective_sync_entry', 'pre_collective_sync_return',
            'collective_host_entry', 'collective_host_return',
            'post_collective_sync_entry', 'post_collective_sync_return'])
        self.assertEqual([r['device_completion_observed'] for r in records],
                         [False, True, False, False, False, True])

    def test_pre_fence_failure_prevents_collective(self):
        calls, records = self.run_control(1)
        self.assertEqual(calls, ['sync'])
        self.assertEqual(records[-1]['stage'], 'pre_collective_sync_entry')

    def test_collective_failure_prevents_post_fence(self):
        calls, records = self.run_control('reduce')
        self.assertEqual(calls, ['sync', 'reduce'])
        self.assertEqual(records[-1]['stage'], 'collective_host_entry')

    def test_post_fence_failure_does_not_claim_completion(self):
        calls, records = self.run_control(3)
        self.assertEqual(calls, ['sync', 'reduce', 'sync'])
        self.assertEqual(records[-1]['stage'], 'post_collective_sync_entry')
        self.assertFalse(records[-1]['device_completion_observed'])


if __name__ == '__main__':
    unittest.main()
