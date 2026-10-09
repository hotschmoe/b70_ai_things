#!/usr/bin/env python3
"""HC descriptors from the fresh exact-artifact inventory and payload receipt."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
F04=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009')
INVENTORY=F04/'current-gguf-inventory.json'
PAYLOADS=F04/'native-hc-source-tensors.json'


def load_records():
    inventory=json.loads(INVENTORY.read_bytes())
    payload=json.loads(PAYLOADS.read_bytes())
    lock=json.loads((ROOT/'strata/flash-next/model-lock.json').read_bytes())
    assert inventory['inventory_complete'] and not inventory['errors'] and payload['passed']
    assert payload['inventory_sha256']==hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    assert inventory['source_revision']==payload['model_revision']==lock['revision']
    audited={item['name']:item for item in payload['tensors']}
    expected=[(0,[10240],40960),(8,[10240,320],3481600),
              (8,[320,10240],3481600),(0,[10240,4],163840)]
    records=[]
    for file in inventory['files']:
        if 'UD-Q4_K_XL-' not in Path(file['path']).name:
            continue
        locked=next(item for item in lock['files'] if Path(item['path']).name==Path(file['path']).name)
        assert file['size_bytes']==locked['size']
        for tensor in file['tensors']:
            name=tensor['name']
            if not re.fullmatch(r'(?:blk\.[0-9]+\.hc_(?:attn|ffn)_|output_hc_)(?:norm|down|up|inject)\.weight',name):
                continue
            role=['norm','down','up','inject'].index(name.rsplit('_',1)[1].split('.')[0])
            dtype,shape,size=expected[role]
            assert tensor['type_id']==dtype and tensor['shape_ggml_order']==shape and tensor['packed_bytes']==size
            absolute=file['tensor_data_offset']+tensor['relative_offset']
            assert tensor['absolute_offset']==absolute and absolute+size<=file['size_bytes']
            audit=audited[name]
            assert audit['shard']==file['path'] and audit['absolute_offset']==absolute
            assert audit['shape']==shape and audit['type_id']==dtype and audit['bytes']==size
            records.append(dict(name=name,role=role,type=dtype,shape=shape,bytes=size,
                                data_start=file['tensor_data_offset'],relative_offset=tensor['relative_offset'],
                                absolute_offset=absolute,file_size=file['size_bytes'],
                                shard=Path(file['path']).name,shard_locked_sha256=locked['sha256'],
                                tensor_sha256=audit['sha256']))
    names={item['name'] for item in records}
    expected_names={f'blk.{layer}.hc_{half}_{role}.weight' for layer in range(48)
                    for half in ['attn','ffn'] for role in ['norm','down','up','inject']}
    expected_names|={f'output_hc_{role}.weight' for role in ['norm','down','up']}
    assert len(records)==len(names)==387 and names==expected_names==set(audited)
    return records


def fixture_header():
    lines=['#pragma once', 'struct ArtifactHc { const char* name; int role,type,rank; '
           'unsigned long long ne0,ne1,bytes,data_start,relative,absolute,file_size; };',
           'constexpr ArtifactHc artifact_hc[] = {']
    for r in load_records():
        shape=r['shape']
        vals=[r['role'],r['type'],len(shape),shape[0],shape[1] if len(shape)>1 else 1,
              r['bytes'],r['data_start'],r['relative_offset'],r['absolute_offset'],r['file_size']]
        lines.append('    {"'+r['name']+'",'+','.join(str(x)+'ull' if i>=3 else str(x)
                                                   for i,x in enumerate(vals))+'},')
    return '\n'.join(lines+['};',''])


if __name__=='__main__':
    records=load_records()
    receipt=dict(scope='F04 exact-artifact HC descriptor and payload-receipt cross-check; no new weight read',
                 inventory=str(INVENTORY),inventory_sha256=hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
                 payload_receipt=str(PAYLOADS),payload_receipt_sha256=hashlib.sha256(PAYLOADS.read_bytes()).hexdigest(),
                 records=records)
    (ROOT/'strata/flash-next/native-hc-artifact-descriptors.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('PASS actual artifact:387 HC tensors,48 layers plus head; dtype/rank/shape/bytes/offset bounds and payload receipts')
