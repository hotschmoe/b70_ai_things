import contextlib
import ast
import io
import json
import os
from pathlib import Path
import types
import unittest

import embedding_trace as trace
import prepare_embedding_trace as prepare


class Tensor:
    shape = (8192, 5120)
    dtype = 'torch.float16'
    device = 'xpu:0'

    def stride(self):
        return (5120, 1)

    def numel(self):
        return 8192 * 5120

    def element_size(self):
        return 2


class TraceTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('B70_EMBED_SOURCE'), 'requires exact exported source')
    def test_actual_patched_forward(self):
        tree = ast.parse(prepare.patch(Path(os.environ['B70_EMBED_SOURCE']).read_bytes()))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'VocabParallelEmbedding')
        forward = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')
        for rank in (0, 1):
            for route, scattered, tp in [('tp', False, 2), ('attn', False, 2),
                                         ('none', True, 2), ('none', False, 1)]:
                with self.subTest(rank=rank, route=route, tp=tp):
                    calls = []
                    env = dict(_b70_trace=trace, maybe_detect_oob=lambda *a: None,
                               get_tp_group=lambda: types.SimpleNamespace(rank_in_group=rank),
                               get_attn_tp_context=lambda: types.SimpleNamespace(input_scattered=scattered),
                               attn_tp_all_reduce=lambda x: calls.append('attn') or x,
                               tensor_model_parallel_all_reduce=lambda x: calls.append('tp') or x)
                    exec(compile(ast.Module(body=[forward], type_ignores=[]), 'patched', 'exec'), env)
                    tensor = Tensor()
                    owner = types.SimpleNamespace(num_embeddings=100, tp_size=tp,
                                                  use_attn_tp_group=route == 'attn',
                                                  _embed_local_shard=lambda x: tensor)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertIs(env['forward'](owner, object()), tensor)
                    records = [json.loads(x.split(' ', 1)[1]) for x in output.getvalue().splitlines()]
                    self.assertEqual(calls, [] if route == 'none' else [route])
                    self.assertEqual(len(records), 1 if route == 'none' else 3)
                    self.assertTrue(all(x['rank'] == rank for x in records))

    def test_metadata_order_and_identity(self):
        tensor = Tensor()
        output = io.StringIO()
        calls = []
        def collective(value):
            calls.append(value)
            return value
        with contextlib.redirect_stdout(output):
            sequence = trace.producer(tensor, 1)
            result = trace.reduce_call(collective, tensor, 1, sequence, 'tp')
        records = [json.loads(line.split(' ', 1)[1]) for line in output.getvalue().splitlines()]
        self.assertIs(result, tensor)
        self.assertEqual(calls, [tensor])
        self.assertEqual([r['stage'] for r in records], [
            'producer_host_return', 'collective_host_entry', 'collective_host_return'])
        for record in records:
            self.assertEqual(record['bytes'], 83886080)
            self.assertEqual(record['stride'], [5120, 1])
            self.assertEqual(record['rank'], 1)
            self.assertEqual(record['sequence'], sequence)
            self.assertFalse(record['device_completion_observed'])

    def test_failure_has_entry_without_return(self):
        output = io.StringIO()
        def failed(tensor):
            raise RuntimeError('device lost')
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'device lost'):
            trace.reduce_call(failed, Tensor(), 0, 12, 'tp')
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn('collective_host_entry', output.getvalue())

    def test_reject_other_source(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            prepare.patch(b'unknown source')


if __name__ == '__main__':
    unittest.main()
