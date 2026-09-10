#!/usr/bin/env python3
"""Read-only candidate AST ordering and gather checks; no backend import."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    source = args.candidate.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert digest == '7299b4cfabc447b15b66a5fcaf5bb858a0783fc46340d561f292f5f96e99e789'
    tree = ast.parse(source)
    update = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_update_states')
    wait = next(n for n in update.body if isinstance(n, ast.If) and
                ast.unparse(n.test) == 'self.num_accepted_tokens_event is not None')
    assert ast.unparse(wait.body[0]) == 'self.num_accepted_tokens_event.synchronize()'
    positions = {}
    for node in ast.walk(update):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                'add_request', 'condense', '_may_reorder_batch'):
            positions.setdefault(node.func.attr, []).append(node.lineno)
    assert set(positions) == {'add_request', 'condense', '_may_reorder_batch'}
    assert all(wait.lineno < line for lines in positions.values() for line in lines)
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                  ast.unparse(n.test) == 'needs_cpu_accepted_counts')
    code = compile(ast.Module(body=[branch], type_ignores=[]), str(args.candidate), 'exec')
    rows = []
    for counts, previous in [([4, 1], [1, 0]), ([2, 4], [1, 2]), ([1, 4], [-1, 1])]:
        events = []
        output = np.zeros(4, dtype=np.int32)
        runner = NS(num_accepted_tokens_event=NS(synchronize=lambda: events.append('wait')),
                    num_accepted_tokens=NS(np=output, copy_to_gpu=lambda: events.append('copy')),
                    input_batch=NS(num_accepted_tokens_cpu=np.array(counts)),
                    use_async_scheduling=True, prev_positions=NS(np=np.array(previous)))
        exec(code, dict(self=runner, needs_cpu_accepted_counts=True, num_reqs=2,
                        np=np, prev_req_id_to_index={'dummy': 0}))
        assert output.tolist() == counts + [1, 1]
        assert events == ['wait', 'copy']
        rows.append(dict(current_counts=counts, old_positions=previous,
                         result=output.tolist(), events=events))
    print(json.dumps(dict(candidate_sha256=digest, early_wait_line=wait.lineno,
                          row_moves=positions, actual_candidate_gather_cases=rows,
                          result='PASS; CPU contract only, no async GPU race qualification'), indent=2))


if __name__ == '__main__':
    main()
