#!/usr/bin/env python3
"""Run only under cpu-run; two bounded real tensor reads, no payload copies."""
from pathlib import Path
import hashlib,json,time
p=Path(__file__).parent
assert not (p/'weight-identity.json').exists(),'preserve existing identity'
prior=json.loads((p.parent/'iq3s-dense-next-s63-prep/iq3s-tensor-descriptors.json').read_text())
model=Path(prior['path']);st=model.stat();assert st.st_size==prior['size']
names=['blk.0.attn_gate.weight','blk.1.ssm_out.weight'];rows=[];t0=time.monotonic()
with model.open('rb') as f:
 for name in names:
  row=dict(next(x for x in prior['rows'] if x['tensor']==name));f.seek(row['absolute_offset']);b=f.read(row['bytes']);assert len(b)==row['bytes']
  h=0xcbf29ce484222325
  for v in b:h=((h^v)*0x100000001b3)&((1<<64)-1)
  row.update(sha256=hashlib.sha256(b).hexdigest(),fnv1a64=f'{h:016x}');rows.append(row)
identity={'path':str(model),'locked_full_model_sha256_not_rehashed':prior['locked_sha256_not_rehashed'],
 'file_size':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'C':6,'tensors':rows,
 'payload_bytes_read':sum(x['bytes'] for x in rows),'payload_bytes_copied':0,'seconds':time.monotonic()-t0,
 'scope':'two finite tensor reads under cpu-run, no GPU or full-shard hash'}
(p/'weight-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
