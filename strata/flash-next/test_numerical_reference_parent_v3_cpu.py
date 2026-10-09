#!/usr/bin/env python3
"""Reference parent preserves frozen gates and exact failed two-page views."""
import ast
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import run_source_upload_oracle_full_v2 as upload

HERE = Path(__file__).resolve().parent


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def main():
    old = functions(HERE / 'qualify_layer0_numerical_v2.py')
    new = functions(HERE / 'qualify_layer0_numerical_reference_v3.py')
    for name in ['stat_signature', 'full_buffered_identity', 'finalizable', 'source_plan_binding', 'prepared_chain_binding']:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    source_watch = next(n for n in ast.walk(new['main']) if isinstance(n, ast.FunctionDef) and n.name == 'source_watch')
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        source = out / 'synthetic-source'
        good = b'\x45' * 4096
        source.write_bytes(good * 2)
        digest = hashlib.sha256(good).hexdigest()
        calls = []
        namespace = {'shards': [source] * 4, 'out': out, 'Path': Path, 'plan': {'model_identity': {'path': str(out / 'unused.json')}}, 'lock': {},
                     'guarded_pages': upload.guarded_pages,
                     'ctrl': SimpleNamespace(verify_model_identity=lambda *args: calls.append(args) or {'passed': True})}
        module = ast.Module(body=[source_watch], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<actual source_watch>', 'exec'), namespace)
        with patch.object(upload.watchdog, 'KNOWN_PAGES', ((0, digest), (4096, digest))):
            assert namespace['source_watch']('good')['reference_identity']['passed']
            assert len(calls) == 1
            raw = bytearray(good * 2)
            raw[4096 + 2796] ^= 0x20
            source.write_bytes(raw)
            try:
                namespace['source_watch']('bad')
            except ValueError:
                pass
            else:
                raise AssertionError('Bad second page accepted')
            assert len(calls) == 1, 'Legacy check must not replace failed views'
            assert next(out.glob('bad-*-buffered-page-0.raw')).read_bytes() == good
            assert next(out.glob('bad-*-buffered-page-4096.raw')).read_bytes() == bytes(raw[4096:])
    print('PASS actual reference source_watch second-page failure/preservation and frozen lifecycle/hash gates; CPU only')


if __name__ == '__main__':
    main()
