#!/usr/bin/env python3
"""Freeze bounded HC/PLE upload SHA expectations from actual pinned source receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True,type=Path)
    p.add_argument('--hc-receipt',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() or a.receipt.exists():p.error('New output/receipt paths required')
    inventory=json.loads(a.inventory.read_text());hc=json.loads(a.hc_receipt.read_text())
    if not inventory['inventory_complete'] or inventory['receipt_status']!='verified' or hc['model_revision']!='766911a6b7369840a91dbcd95f9f997acaab6cd6':raise RuntimeError('Complete selected model source receipts required')
    names={'blk.1.ple_key.weight','blk.1.ple_value.weight','blk.1.ple_conv1d.weight'}
    selected=list(hc['tensors'])
    if len(selected)!=387:raise RuntimeError('Expected387 source HC images')
    for f in inventory['files']:
        if '/UD-Q4_K_XL/' not in f['path']:continue
        with Path(f['path']).open('rb') as h:
            for t in f['tensors']:
                if t['name'] not in names:continue
                h.seek(t['absolute_offset']);raw=h.read(t['packed_bytes']);assert len(raw)==t['packed_bytes']
                selected.append(dict(name=t['name'],shard=f['path'],type_id=t['type_id'],bytes=t['packed_bytes'],absolute_offset=t['absolute_offset'],sha256=hashlib.sha256(raw).hexdigest()))
    if len(selected)!=390:raise RuntimeError('Expected387 HC plus3 PLE images')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='ascii') as h:
        h.write('# name shard_basename type absolute_offset bytes source_sha256\n')
        for row in sorted(selected,key=lambda r:r['name']):h.write('\t'.join(map(str,[row['name'],Path(row['shard']).name,row['type_id'],row['absolute_offset'],row['bytes'],row['sha256']]))+'\n')
    receipt={'CONFIG':'Actual pinned selected sourceHC receipt plus bounded originalPLE source bytes; no GPU, conversion or103GB scan','COMMAND':['prepare_source_upload_roster','--inventory',str(a.inventory),'--hc-receipt',str(a.hc_receipt)],'RESULT':{'model_revision':hc['model_revision'],'inventory_sha256':sha(a.inventory),'HC_source_receipt_sha256':sha(a.hc_receipt),'roster_sha256':sha(a.output),'images':len(selected),'image_bytes':sum(row['bytes'] for row in selected),'rows':selected,'full_shard_hash_reverified':False},'VERDICT':'Frozen expected source SHA256 for whole HC/PLE source upload oracle only; model/route math and GPU uploads unqualified'}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
    print(json.dumps({'images':len(selected),'roster':str(a.output),'receipt':str(a.receipt)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
