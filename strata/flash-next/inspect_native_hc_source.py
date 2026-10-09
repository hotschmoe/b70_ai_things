#!/usr/bin/env python3
"""Read and receipt actual selected HC source tensors; no GPU or model conversion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['inventory_complete'] and inventory['receipt_status'] == 'verified'
    assert inventory['source_revision'] == '766911a6b7369840a91dbcd95f9f997acaab6cd6'
    shapes = {'norm': ([10240], 0, 40960), 'down': ([10240, 320], 8, 3481600),
              'up': ([320, 10240], 8, 3481600), 'inject': ([10240, 4], 0, 163840)}
    records = []
    for file in inventory['files']:
        path = Path(file['path'])
        if 'UD-Q4_K_XL' not in path.name:
            continue
        assert file['receipt_identity_matches'] and path.stat().st_size == file['size_bytes']
        with path.open('rb') as handle:
            for tensor in file['tensors']:
                name = tensor['name']
                if not ('.hc_attn_' in name or '.hc_ffn_' in name or name.startswith('output_hc_')):
                    continue
                role = name.rsplit('_', 1)[-1].split('.')[0]
                shape, dtype, size = shapes[role]
                assert tensor['shape_ggml_order'] == shape and tensor['type_id'] == dtype
                assert tensor['packed_bytes'] == size
                offset = tensor['absolute_offset']
                assert 0 <= offset <= path.stat().st_size - size
                handle.seek(offset)
                data = handle.read(size)
                assert len(data) == size
                record = dict(name=name, shard=str(path), shape=shape, type_id=dtype, bytes=size,
                              absolute_offset=offset, sha256=hashlib.sha256(data).hexdigest())
                if dtype == 0:
                    values = [x[0] for x in struct.iter_unpack('<f', data)]
                    assert all(math.isfinite(v) for v in values)
                    record.update(min=min(values), max=max(values), mean=sum(values)/len(values),
                                  bf16_nonrepresentable=sum((x[0] & 65535) != 0 for x in struct.iter_unpack('<I', data)))
                records.append(record)
    names = {r['name'] for r in records}
    wanted = {f'blk.{layer}.hc_{half}_{role}.weight' for layer in range(48)
              for half in ['attn', 'ffn'] for role in shapes}
    wanted.update('output_hc_' + role + '.weight' for role in ['norm', 'down', 'up'])
    assert names == wanted and len(records) == len(wanted) == 387
    result = dict(scope='Actual HC source tensor identity/type/shape/value audit; no upload or GPU arithmetic',
                  inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                  prior_full_file_hash_receipt=inventory['hash_receipt'], full_shard_hash_reverified=False,
                  model_revision=inventory['source_revision'], tensors=records, passed=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, tensor_count=len(records),
                         injection_nonrepresentable=sum(r.get('bf16_nonrepresentable', 0) for r in records if '_inject.' in r['name']),
                         receipt=str(args.output))))


if __name__ == '__main__':
    main()
