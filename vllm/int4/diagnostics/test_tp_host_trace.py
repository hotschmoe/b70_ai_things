"""CPU-only startup trace fixtures; no Torch/backend import or GPU access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HOOK = Path(__file__).resolve().parents[2] / 'fp8/kv_hooks/b70_tp_host_trace.py'


class Tensor:
    shape = (32768, 5120)
    dtype = 'float16'
    device = 'xpu:1'

    def stride(self):
        return (5120, 1)

    def cpu(self):
        raise AssertionError('No tensor copy permitted')

    def item(self):
        raise AssertionError('No tensor value permitted')


class HostTraceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('host_trace', HOOK)
        self.hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hook)
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'B70_TP_HOST_TRACE_DIR': self.directory.name})
        self.env.start()
        self.group = types.SimpleNamespace(rank_in_group=1, world_size=2,
                                          unique_name='tp:0', use_custom_op_call=True)
        parallel = types.ModuleType('vllm.distributed.parallel_state')
        parallel.get_tp_group = lambda: self.group
        self.modules = patch.dict(sys.modules, {parallel.__name__: parallel})
        self.modules.start()

        class Communicator:
            unique_name = 'tp:0'
            world_size = 2
            rank_in_group = 1
            global_rank = 1

            def all_reduce(self, value):
                return value

        self.communicator = Communicator()
        comm = self.communicator

        class Runner:
            def _dummy_run(self, num_tokens, is_profile=False,
                           is_graph_capturing=False, nested=False, fail=False):
                comm.all_reduce(Tensor())
                if nested:
                    self._dummy_run(1, is_graph_capturing=True)
                if fail:
                    raise RuntimeError('fixture')
                return (Tensor(), Tensor())

        # Fake class files are not backend sources. Test the real source gate
        # separately; disable only this install-time check for boundary fixtures.
        with patch.object(self.hook, 'verify'):
            self.hook.install_communicator(Communicator)
            self.hook.install_runner(Runner)
        self.runner = Runner()

    def tearDown(self):
        if self.hook._fd is not None:
            os.close(self.hook._fd)
        self.modules.stop()
        self.env.stop()
        self.directory.cleanup()

    def rows(self):
        paths = list(Path(self.directory.name).glob('*.jsonl'))
        return [json.loads(line) for path in paths for line in path.read_text().splitlines()]

    def test_nested_profile_and_capture_metadata(self):
        self.runner._dummy_run(32768, is_profile=True, nested=True)
        rows = self.rows()
        returns = [row for row in rows if row['event'] == 'dummy_return']
        self.assertEqual([row['phase'] for row in returns], ['capture', 'profile'])
        for row in returns:
            self.assertEqual(row['counts'], {'all_reduce_enter': 1, 'all_reduce_return': 1})
        enters = [row for row in rows if row['event'] == 'collective_enter']
        self.assertEqual(len(enters), 1)
        self.assertEqual(enters[0]['input']['shape'], [32768, 5120])
        self.assertEqual(enters[0]['communicator']['global_rank'], 1)
        self.assertIsNone(self.hook._local.scope)

    def test_exception_and_logging_failure_restore_scope(self):
        with self.assertRaises(RuntimeError):
            self.runner._dummy_run(1, is_profile=True, fail=True)
        self.assertIsNone(self.hook._local.scope)
        with patch.object(self.hook, 'emit', side_effect=OSError('write fixture')):
            with self.assertRaises(OSError):
                self.runner._dummy_run(1, is_profile=True)
        self.assertIsNone(self.hook._local.scope)

    def test_other_group_skipped_and_nonopaque_route_rejected(self):
        self.communicator.unique_name = 'dp:0'
        self.runner._dummy_run(1, is_profile=True)
        returned = [row for row in self.rows() if row['event'] == 'dummy_return'][0]
        self.assertEqual(returned['counts'], {'non_tp_communicator_skipped': 1})
        self.assertFalse(any(row['event'] == 'collective_enter' for row in self.rows()))
        self.group.use_custom_op_call = False
        with self.assertRaises(RuntimeError):
            self.runner._dummy_run(1, is_profile=True)

    def test_bounded_event_stream_marks_incomplete(self):
        with patch.dict(os.environ, {'B70_TP_HOST_TRACE_MAX_EVENTS': '2'}):
            self.runner._dummy_run(1, is_profile=True)
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]['event'], 'truncated')
        self.assertTrue(rows[-1]['counts_incomplete'])

    def test_source_gate_rejects_unreviewed_bytes(self):
        source = Path(__file__).read_text().rstrip('\n') + '\n'
        self.hook.EXPECTED['runner'] = hashlib.sha256(source.encode()).hexdigest()
        self.hook.verify(Tensor, 'runner')
        with patch.dict(os.environ, {'B70_TP_HOST_TRACE_PROFILE': 'phase-mrv1'}):
            with self.assertRaises(RuntimeError):
                self.hook.verify(Tensor, 'runner')
            with patch.object(self.hook, 'MRV1_RUNNER_SHA256', self.hook.EXPECTED['runner']):
                self.hook.verify(Tensor, 'runner')
        with patch.dict(os.environ, {'B70_TP_HOST_TRACE_PROFILE': 'unreviewed'}):
            with self.assertRaises(RuntimeError):
                self.hook.verify(Tensor, 'runner')
        self.hook.EXPECTED['runner'] = '0' * 64
        with self.assertRaises(RuntimeError):
            self.hook.verify(Tensor, 'runner')


if __name__ == '__main__':
    unittest.main()
