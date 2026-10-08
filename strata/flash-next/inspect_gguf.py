#!/usr/bin/env python3
"""Bounded CPU-only GGUF header inventory; never reads tensor payloads.

Full-file SHA256 identities are referenced from the downloader receipt, not
recomputed here. Payload sizes describe packed file storage, not device
allocation (padding, conversion, KV and scratch are excluded).
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GGUF = Path('/mnt/vm_8tb/github/llama.cpp-flashnext/gguf-py')
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
SPLIT = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


class HeaderReader:
    def __init__(self, stream, limit):
        self.stream = stream
        self.limit = limit
        self.header_hash = hashlib.sha256()
        self.value_hash = None
        self.endian = '<'

    def read(self, size):
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError('header exceeds byte limit')
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError('truncated GGUF header')
        self.header_hash.update(data)
        if self.value_hash is not None:
            self.value_hash.update(data)
        return data

    def number(self, fmt):
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, capture=True):
        length = self.number('Q')
        if length > self.limit:
            raise ValueError('GGUF string exceeds byte limit')
        if capture:
            # Names and retained values are bounded independently of header size.
            if length > 1024 * 1024:
                raise ValueError('retained GGUF string exceeds 1 MiB')
            return self.read(length).decode('utf-8')
        digest = hashlib.sha256()
        while length:
            chunk = self.read(min(length, 65536))
            digest.update(chunk)
            length -= len(chunk)
        return {'utf8_sha256': digest.hexdigest()}

    def value(self, kind, capture=True, depth=0):
        if kind in SCALARS:
            return self.number(SCALARS[kind])
        if kind == 8:
            return self.string(capture=capture)
        if kind == 9:
            if depth >= 4:
                raise ValueError('metadata array nesting exceeds limit')
            subtype, count = self.number('I'), self.number('Q')
            if count > 2000000:
                raise ValueError('metadata array count exceeds limit')
            values = [] if capture and count <= 128 else None
            for _ in range(count):
                value = self.value(subtype, capture=values is not None, depth=depth + 1)
                if values is not None:
                    values.append(value)
            result = {'element_type': subtype, 'count': count}
            if values is not None:
                result['values'] = values
            return result
        raise ValueError('unknown GGUF metadata type: ' + str(kind))


def category(name, path):
    if 'mtp' in path.stem.lower() or re.search(r'(^|\.)(nextn|mtp)[_.]', name):
        return 'mtp'
    if 'per_layer_token_embd' in name or 'ngram_embedding' in name:
        return 'ple'
    if re.search(r'ffn_(up|gate|down|gate_up)_exps\.', name):
        return 'routed_experts'
    if re.search(r'ffn_.*shexp', name):
        return 'shared_experts'
    return 'dense'


def inspect_file(path, constants, limit):
    before = path.stat()
    with path.open('rb', buffering=0) as stream:
        reader = HeaderReader(stream, limit)
        if reader.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        version_bytes = reader.read(4)
        version = struct.unpack('<I', version_bytes)[0]
        if version not in (2, 3):
            version = struct.unpack('>I', version_bytes)[0]
            reader.endian = '>'
        if version not in (2, 3):
            raise ValueError('only GGUF v2/v3 supported')
        tensor_count, metadata_count = reader.number('Q'), reader.number('Q')
        if tensor_count > 1000000 or metadata_count > 100000:
            raise ValueError('GGUF tensor/metadata count exceeds limit')
        metadata = {}
        tokenizer_hash = hashlib.sha256()
        tokenizer_fields = 0
        chat_hashes = {}
        for _ in range(metadata_count):
            key, kind = reader.string(), reader.number('I')
            if key in metadata:
                raise ValueError('duplicate metadata key: ' + key)
            reader.value_hash = hashlib.sha256()
            value = reader.value(kind, capture=not key.startswith('tokenizer.'))
            digest = reader.value_hash.hexdigest()
            reader.value_hash = None
            metadata[key] = {'type': kind, 'value': value, 'encoded_value_sha256': digest}
            if key.startswith('tokenizer.'):
                tokenizer_hash.update(json.dumps([key, kind, digest], ensure_ascii=True,
                                                 separators=(',', ':')).encode('ascii') + b'\n')
                tokenizer_fields += 1
            if key.startswith('tokenizer.chat_template'):
                chat_hashes[key] = value
        alignment = metadata.get('general.alignment', {}).get('value', 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        tensors = []
        names = set()
        for _ in range(tensor_count):
            name, ndims = reader.string(), reader.number('I')
            if name in names or not 1 <= ndims <= 4:
                raise ValueError('duplicate tensor name or invalid dimension count')
            names.add(name)
            shape = [reader.number('Q') for _ in range(ndims)]
            kind, offset = reader.number('I'), reader.number('Q')
            quant_type = constants['GGMLQuantizationType'](kind)
            block, block_bytes = constants['GGML_QUANT_SIZES'][quant_type]
            if any(dim == 0 for dim in shape) or shape[0] % block:
                raise ValueError('tensor has invalid quantized row shape: ' + name)
            size = math.prod(shape) // block * block_bytes
            match = re.search(r'(?:^|\.)blk\.(\d+)\.', name)
            tensors.append({'name': name, 'type': quant_type.name, 'type_id': kind,
                            'shape_ggml_order': shape, 'elements': math.prod(shape),
                            'packed_bytes': size, 'relative_offset': offset,
                            'layer': int(match[1]) if match else None,
                            'category': category(name, path)})
        header_bytes = stream.tell()
        data_offset = (header_bytes + alignment - 1) // alignment * alignment
        previous_end = 0
        for tensor in sorted(tensors, key=lambda t: t['relative_offset']):
            offset = tensor['relative_offset']
            if offset % alignment or offset < previous_end:
                raise ValueError('misaligned/overlapping tensor: ' + tensor['name'])
            previous_end = offset + tensor['packed_bytes']
            tensor['absolute_offset'] = data_offset + offset
            if data_offset + previous_end > before.st_size:
                raise ValueError('tensor extends past EOF: ' + tensor['name'])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed during header inspection')
    return {'path': str(path), 'size_bytes': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'gguf_version': version,
            'endian': 'little' if reader.endian == '<' else 'big',
            'header_bytes_read': header_bytes, 'header_sha256': reader.header_hash.hexdigest(),
            'tensor_data_offset': data_offset, 'metadata': metadata, 'tensors': tensors,
            'tokenizer_metadata_sha256': tokenizer_hash.hexdigest() if tokenizer_fields else None,
            'tokenizer_hash_contract': 'ordered JSON [key,type,SHA256(encoded GGUF value)] plus LF',
            'chat_template_hashes': chat_hashes}


def summarize(files):
    groups = defaultdict(list)
    for file in files:
        path = Path(file['path'])
        match = SPLIT.match(path.name)
        key = str(path.parent / match[1]) if match else str(path)
        groups[key].append(file)
    result = []
    for key, shards in sorted(groups.items()):
        issues, categories, layers, types = [], defaultdict(int), defaultdict(lambda: defaultdict(int)), defaultdict(int)
        expected_counts, indices, declared_totals, seen_names = set(), [], set(), set()
        for shard in shards:
            meta = {k: v['value'] for k, v in shard['metadata'].items()}
            match = SPLIT.match(Path(shard['path']).name)
            count = meta.get('split.count', 1)
            index = meta.get('split.no', 0)
            expected_counts.add(count)
            indices.append(index)
            if match and (int(match[2]) != index + 1 or int(match[3]) != count):
                issues.append('filename disagrees with split metadata: ' + shard['path'])
            if 'split.tensors.count' in meta:
                declared_totals.add(meta['split.tensors.count'])
            for tensor in shard['tensors']:
                if tensor['name'] in seen_names:
                    issues.append('duplicate tensor across shards: ' + tensor['name'])
                seen_names.add(tensor['name'])
                size, cat = tensor['packed_bytes'], tensor['category']
                categories[cat] += size
                layers[str(tensor['layer']) if tensor['layer'] is not None else 'global'][cat] += size
                types[tensor['type']] += size
        expected = next(iter(expected_counts)) if len(expected_counts) == 1 else None
        if (not isinstance(expected, int) or not 1 <= expected <= 100000 or
                sorted(indices) != list(range(expected))):
            issues.append('missing, duplicate, or inconsistent split indices/counts')
        if declared_totals and (len(declared_totals) != 1 or next(iter(declared_totals)) != len(seen_names)):
            issues.append('split.tensors.count disagrees with observed tensor count')
        # Split producers may emit tokenizer metadata only in shard zero.
        tokenizer_hashes = {s['tokenizer_metadata_sha256'] for s in shards if s['tokenizer_metadata_sha256']}
        result.append({'group': key, 'split_complete': not issues, 'issues': issues,
                       'expected_shards': expected, 'observed_shards': len(shards),
                       'tensor_count': len(seen_names), 'packed_bytes_by_category': dict(categories),
                       'packed_bytes_by_layer': dict(sorted(layers.items())),
                       'packed_bytes_by_type': dict(types), 'mixed_quant_types': len(types) > 1,
                       'tokenizer_hashes_present': sorted(tokenizer_hashes)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path,
                        help='GGUF files or directories; omitted: all GGUFs in the model lock')
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('model-lock.json'))
    parser.add_argument('--receipt', type=Path, help='downloader SHA256 receipt; referenced, not reverified')
    parser.add_argument('--gguf-py', type=Path, default=DEFAULT_GGUF,
                        help='fresh llama.cpp gguf-py directory (imports constants only)')
    parser.add_argument('--max-header-mib', type=int, default=128)
    parser.add_argument('--output', type=Path, help='JSON output (default stdout)')
    args = parser.parse_args()
    constants_path = args.gguf_py / 'gguf/constants.py'
    constants = runpy.run_path(str(constants_path))
    lock = json.loads(args.lock.read_text()) if args.lock.exists() else {}
    root = REPO / lock.get('destination', '.')
    entries = {str((root / entry['path']).resolve()): entry for entry in lock.get('files', [])}
    paths = []
    if args.paths:
        for path in args.paths:
            paths.extend(path.rglob('*.gguf') if path.is_dir() else [path])
    else:
        paths = [Path(path) for path in entries if path.endswith('.gguf')]
    receipt = json.loads(args.receipt.read_text()) if args.receipt else {}
    verified = {row['path']: row for row in receipt.get('verified', [])}
    files, errors = [], []
    for path in sorted({p.resolve() for p in paths}):
        try:
            file = inspect_file(path, constants, args.max_header_mib * 1024 * 1024)
            entry = entries.get(str(path))
            receipt_match = False
            if entry:
                row = verified.get(entry['path'], {})
                receipt_match = (receipt.get('repo') == lock.get('repo') and
                                 receipt.get('revision') == lock.get('revision') and
                                 Path(receipt.get('destination', '/missing')).resolve() == root.resolve() and
                                 row.get('digest') == entry.get('sha256') and
                                 row.get('size') == entry.get('size') == file['size_bytes'])
                file['publisher_expected_sha256'] = entry.get('sha256')
                file['publisher_expected_size'] = entry['size']
                if entry['size'] != file['size_bytes']:
                    errors.append({'path': str(path), 'error': 'publisher size mismatch'})
            file['receipt_identity_matches'] = receipt_match
            file['full_file_hash_verified_by_this_inspector'] = False
            files.append(file)
        except (OSError, ValueError, KeyError, struct.error) as error:
            errors.append({'path': str(path), 'error': str(error)})
    groups = summarize(files)
    if not paths:
        errors.append({'error': 'no GGUF files selected'})
    revision = subprocess.run(['git', '-C', str(args.gguf_py), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False)
    result = {'schema': 1, 'mode': 'CPU header-only; tensor payloads not read',
              'storage_note': 'Packed file bytes only; excludes device padding, conversion, KV and scratch.',
              'classification_note': 'Name-based; separate mtp files/nextn names classified MTP; no implicit last-block inference.',
              'hash_note': 'Receipt is prior downloader evidence, not a fresh full-file verification.',
              'gguf_constants_source': str(constants_path.resolve()),
              'gguf_constants_sha256': hashlib.sha256(constants_path.read_bytes()).hexdigest(),
              'llama_cpp_revision': revision.stdout.strip() if revision.returncode == 0 else None,
              'model_lock': str(args.lock.resolve()) if lock else None,
              'source_repo': lock.get('repo'), 'source_revision': lock.get('revision'),
              'hash_receipt': str(args.receipt.resolve()) if args.receipt else None,
              'receipt_status': receipt.get('status'), 'files': files, 'groups': groups,
              'errors': errors,
              'inventory_complete': not errors and all(g['split_complete'] for g in groups)}
    output = json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='ascii')
    else:
        sys.stdout.write(output)
    return 0 if result['inventory_complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
