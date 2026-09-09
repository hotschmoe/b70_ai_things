#!/usr/bin/env python3
"""Build a fail-closed Python overlay from the exact preserved image source."""
import argparse
import hashlib
import json
from pathlib import Path

BASE_SHA256 = 'c2c8d349cc9ae84b1c7e8fd1a8de6bc45c495302950b64b145023cb2580215c0'


def patch(source):
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError('embedding source identity mismatch')
    text = source.decode('ascii')
    replacements = {
        'import logging\n': 'import logging\nimport b70_embedding_trace as _b70_trace\n',
        '        output_parallel = self._embed_local_shard(input_)\n':
        '        output_parallel = self._embed_local_shard(input_)\n'
        '        _b70_sequence = _b70_trace.producer(output_parallel, get_tp_group().rank_in_group)\n',
    }
    for function in ('attn_tp_all_reduce', 'tensor_model_parallel_all_reduce'):
        replacements[f'output_parallel = {function}(output_parallel)'] = (
            'output_parallel = _b70_trace.reduce_call('
            f'{function}, output_parallel, get_tp_group().rank_in_group, _b70_sequence, "{function}")')
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError('expected exactly one patch site: ' + before)
        text = text.replace(before, after)
    compile(text, 'vocab_parallel_embedding.py', 'exec')
    return text.encode('ascii')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    patched = patch(args.source.read_bytes())
    helper = Path(__file__).with_name('embedding_trace.py').read_bytes()
    args.out.mkdir(parents=True, exist_ok=False)
    files = {'vocab_parallel_embedding.py': patched, 'b70_embedding_trace.py': helper}
    for name, data in files.items():
        (args.out / name).write_bytes(data)
    manifest = dict(base_sha256=BASE_SHA256,
                    files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                    completion='Host call boundaries only; no synchronization or device completion evidence',
                    scope='Embedding producer and TP wrapper only; not all model collectives',
                    deployed=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
