#!/usr/bin/env python3
"""CPU interpretation of installed scalar control flow, not a Triton GPU test.

Demonstrates why a blanket src-column > dest-column guard needs semantic review:
normal speculative verification can publish an earlier accepted boundary from
the later running-state scratch column. No state memory is read or written.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS


class Pointer:
    def __init__(self, value):
        self.value = value

    def __add__(self, offset):
        assert offset == 0
        return self


def run(path):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'postprocess_mamba_fused_kernel')
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    copies, stores = [], []
    scope = dict(tl=NS(program_id=lambda axis: 0, load=lambda p: p.value,
                        store=lambda p, v: stores.append(v)),
                 _copy_mamba_state_block=lambda *a: copies.append(
                     dict(src=a[2], dest=a[3], bias=a[4])))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(path), 'exec'), scope)
    func = scope[fn.name]
    kwargs = {n: 0 for n in inspect.signature(func).parameters}
    kwargs.update(num_reqs=1, block_size=16, TEMPORAL_TILES=1,
        HAS_IDX_MAPPING=False, PRECOMPUTED_NEW_COMPUTED=False,
        num_accepted_tokens_ptr=Pointer(3), mamba_state_idx_ptr=Pointer(1),
        num_scheduled_tokens_ptr=Pointer(4), num_computed_tokens_ptr=Pointer(13),
        num_draft_tokens_ptr=Pointer(3))
    func(**kwargs)
    return dict(copies=copies, accepted_count_stores=stores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('guarded_candidate', type=Path)
    args = p.parse_args()
    baseline = run(args.baseline)
    guarded = run(args.guarded_candidate)
    assert baseline['copies'] == [dict(src=1, dest=0, bias=2)]
    assert guarded['copies'] == []
    print(json.dumps(dict(passed=True, baseline=baseline, guarded=guarded,
        verdict='Blanket backward-copy guard suppresses a reachable speculative boundary publication; not qualified as a repair.'), indent=2))


if __name__ == '__main__':
    main()
