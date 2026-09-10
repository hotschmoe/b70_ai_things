#!/usr/bin/env python3
"""Execute installed CPU row moves and accepted-count gather with simulated DMA.

No backend imports, devices, or tensor kernels. DMA landing is simulated at
explicit points; this establishes ordering correctness, not race frequency.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def extract(source, name):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    batch_source = (args.source_root / 'v1/worker/gpu_input_batch.py').read_text()
    runner_source = (args.source_root / 'v1/worker/gpu_model_runner.py').read_text()
    nodes = [extract(batch_source, n) for n in
             ('swap_states', 'condense', '_get_active_token_count')]
    scope = dict(np=np, MoveDirectionality=NS(SWAP='swap', UNIDIRECTIONAL='move'),
                 swap_dict_values=lambda d, a, b: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<installed-gpu-input-batch>', 'exec'), scope)
    gather = next(n for n in ast.walk(ast.parse(runner_source))
                  if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                  and ast.unparse(n.test) ==
                  'self.use_async_scheduling and prev_req_id_to_index')
    gather_code = compile(ast.Module(body=[gather], type_ignores=[]),
                          '<installed-accepted-count-gather>', 'exec')

    def batch(counts, ids):
        b = NS(_req_ids=ids[:], num_reqs=len([x for x in ids if x]),
               req_output_token_ids=[[] for _ in ids],
               spec_token_ids=[[] for _ in ids],
               req_id_to_index={r: i for i, r in enumerate(ids) if r},
               use_replayssm=False, is_pooling_model=False,
               token_ids_cpu=np.zeros((len(ids), 4), dtype=np.int32),
               is_token_ids=np.ones((len(ids), 4), dtype=bool),
               req_prompt_embeds={}, generators={}, bad_words_token_ids={},
               allowed_token_ids_mask_cpu_tensor=None,
               block_table=NS(swap_row=lambda *a: None, move_row=lambda *a: None))
        for name in ('num_tokens_no_spec', 'num_prompt_tokens',
                     'num_computed_tokens_cpu', 'request_lora_mapping',
                     'temperature_cpu', 'top_p_cpu', 'top_k_cpu',
                     'frequency_penalties_cpu', 'presence_penalties_cpu',
                     'repetition_penalties_cpu'):
            setattr(b, name, np.ones(len(ids), dtype=np.int32))
        b.num_accepted_tokens_cpu = np.array(counts, dtype=np.int32)
        removed = sorted((i for i, x in enumerate(ids) if x is None), reverse=True)
        b.batch_update_builder = NS(removed=removed, moved=[],
            peek_removed=lambda: removed[-1] if removed else None,
            pop_removed=lambda: removed.pop())
        b._get_active_token_count = lambda i: scope['_get_active_token_count'](b, i)
        return b

    rows = []
    for operation in ('unchanged', 'swap', 'condense', 'condense_then_swap'):
        ids = ['A', 'B'] if 'condense' not in operation else [None, 'B', 'C']
        old = [1, 4] if len(ids) == 2 else [1, 2, 4]
        previous = {r: i for i, r in enumerate(ids) if r}
        expected_by_id = {r: old[i] for i, r in enumerate(ids) if r}
        for landing in ('before_moves', 'after_moves'):
            for fixed in (False, True):
                b = batch(old if landing == 'before_moves' or fixed else [1]*len(ids), ids)
                if 'condense' in operation:
                    scope['condense'](b)
                if 'swap' in operation:
                    scope['swap_states'](b, 0, 1)
                if landing == 'after_moves' and not fixed:
                    b.num_accepted_tokens_cpu[:] = old
                expected = [expected_by_id[r] for r in b._req_ids]
                runner = NS(input_batch=b, use_async_scheduling=True,
                    prev_positions=NS(np=np.array([previous[r] for r in b._req_ids])),
                    num_accepted_tokens=NS(np=np.zeros(len(ids), dtype=np.int32)))
                if fixed:
                    runner.num_accepted_tokens.np[:b.num_reqs] = b.num_accepted_tokens_cpu[:b.num_reqs]
                else:
                    exec(gather_code, dict(np=np, self=runner, num_reqs=b.num_reqs,
                                           prev_req_id_to_index=previous))
                actual = runner.num_accepted_tokens.np[:b.num_reqs].tolist()
                should_fail = not fixed and landing == 'before_moves' and 'swap' in operation
                assert (actual != expected) == should_fail, (operation, landing, fixed, actual, expected)
                rows.append(dict(operation=operation, landing=landing, fixed=fixed,
                                 actual=actual, expected=expected, correct=actual == expected))
    print(json.dumps(dict(passed=True, source_sha256={
        'gpu_input_batch.py': hashlib.sha256(batch_source.encode()).hexdigest(),
        'gpu_model_runner.py': hashlib.sha256(runner_source.encode()).hexdigest()},
        cases=rows), indent=2))


if __name__ == '__main__':
    main()
