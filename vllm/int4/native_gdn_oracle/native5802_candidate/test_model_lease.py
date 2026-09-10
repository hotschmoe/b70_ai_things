#!/usr/bin/env python3
"""Execute the actual server CLI/lease branch on CPU, stopping before GPU work."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys

plan = json.loads(Path(sys.argv[1]).read_text())
server = plan['server']
source = Path(server[1])
tree = ast.parse(source.read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
boundary = next(i for i, n in enumerate(main.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'not args.leased')
main.body = main.body[:boundary+1] + [ast.Return(value=ast.Name(id='args', ctx=ast.Load()))]
ast.fix_missing_locations(main)
calls = []
env = dict(argparse=argparse, Path=Path, IMAGE=plan['image'], REPO=source.parents[2],
           os=SimpleNamespace(execv=lambda *a: calls.append(a)), __file__=str(source))
exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), env)
saved = sys.argv
try:
    sys.argv = [str(source), *server[2:]]
    args = env['main']()
    assert args.leased is True and args.card == 0 and args.tensor_parallel_size == 1
    assert server.count('--leased') == 1 and calls == []
    sys.argv = [arg for arg in sys.argv if arg != '--leased']
    args = env['main']()
    assert len(calls) == 1 and calls[0][1][1:3] == ['--card', '0']
finally:
    sys.argv = saved
print('PASS actual server lease branch: --leased once avoids reentry; missing flag reproduces nested gpu-run')
