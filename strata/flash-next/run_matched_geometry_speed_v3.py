#!/usr/bin/env python3
"""Narrow V3 routing of the immutable V2 diagnostic lifecycle."""
import ast
from pathlib import Path
import sys
import run_matched_geometry_speed_v2 as previous
import matched_geometry_c140_diagnostic_v3 as matched

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'strata/flash-next/matched-geometry-speed-source-plan-v3.json'


def sources():
    plan = previous.c1.read(PLAN)
    for row in plan['files']:
        assert previous.c1.sha(ROOT/row['path']) == row['sha256'], 'Matched source closure changed: '+row['path']
    previous.q.source_pins()
    return previous.c1.sha(PLAN)


def routed_main():
    # The original V2 module/functions/globals stay untouched.
    tree = ast.parse(Path(previous.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
    replacements = 0
    for child in ast.walk(node):
        if isinstance(child,ast.Constant) and child.value == 'strata/flash-next/matched_geometry_c140_diagnostic_v2.py':
            child.value = 'strata/flash-next/matched_geometry_c140_diagnostic_v3.py'
            replacements += 1
    assert replacements == 1, 'Exactly one diagnostic controller route required'
    namespace = dict(vars(previous))
    namespace.update(__file__=__file__,__name__=__name__,matched=matched,PLAN=PLAN,sources=sources)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    return namespace['main']


if __name__ == '__main__':
    if sys.flags.optimize: raise ValueError('Optimized Python unsupported')
    routed_main()()
