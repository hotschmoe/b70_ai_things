#!/usr/bin/env python3
"""Future root pre/post check; CPU only. No build or GPU launch is implicit."""
from pathlib import Path
import hashlib,json,sys,platform
p,o=map(Path,sys.argv[1:3]);phase=sys.argv[3] if len(sys.argv)>3 else 'pre'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
host={'kernel':platform.release(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
assert host=={'kernel':'7.1.0-070100-generic','boot_id':'b785982c-4954-4d23-808c-48a7534ce3bd'},host
c=json.loads((p/'root-build-closure.json').read_text())
assert c['host_kernel_binding_distinct'] is True
assert c['device_kernel_names_distinct'] is True
assert c['public_iq_source_sha256']=='acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851'
assert c['public_iq_member_sha256']=='a61c1dcae2783b7b63d23fffed7ccdfa0c79089fee148866cb5f1e111c6bb283'
assert c['image']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
for filename,key in [('build/routed_iq2_leaf_s98','binary_sha256'),('routed_iq2_leaf.cpp','adapter_source_sha256'),('private_iq2_gu.hpp','private_header_sha256')]:assert sha(p/filename)==c[key],filename
# Large immutable inputs are hashed once by the future build; runtime checks
# their exact recorded stat continuity, not a repeated full archive/image read.
stats=json.loads((p/'build/immutable-input-stats.json').read_text())
for filename,v in stats.items():
 s=Path(filename).stat();assert {k:getattr(s,k)for k in v}==v,filename
weight=json.loads((p/'weight-identity.json').read_text());f=Path(weight['compact_file'])
assert f.stat().st_size==weight['compact_bytes']
assert sha(f)==weight['compact_sha256']
assert sha(Path(weight['metadata_path']))==weight['metadata_sha256']
def fnv(b):
 h=0xcbf29ce484222325
 for v in b:h=((h^v)*0x100000001b3)&((1<<64)-1)
 return f'{h:016x}'
with f.open('rb')as s:
 for e in weight['experts']:
  s.seek(e['compact_offset']);b=s.read(e['read_bytes']);assert len(b)==e['read_bytes']
  e.update(slice_sha256=hashlib.sha256(b).hexdigest(),GU_fnv=fnv(b))
d={'phase':phase,'host':host,'binary_sha256':c['binary_sha256'],'source_sha256':c['adapter_source_sha256'],'header_sha256':c['private_header_sha256'],'build_closure_sha256':sha(p/'root-build-closure.json'),'large_input_stat_continuity':True,'weight_identity':weight,'host_binding':True,'device_names':True,'gpu_touched_by_identity':False}
if phase=='post':
 pre=json.loads((o/'identity.json').read_text());assert {k:v for k,v in d.items()if k!='phase'}=={k:v for k,v in pre.items()if k!='phase'}
 for e in [json.loads(l)for l in(o/'leaf.log').read_text().splitlines()if l.startswith('{')]:
  if e.get('type')=='weight_identity':
   wanted=next(x for x in weight['experts']if x['expert']==e['expert']);assert e['GU_fnv']==wanted['GU_fnv']
with(o/('identity.json'if phase=='pre'else'identity-post.json')).open('x',encoding='ascii')as s:json.dump(d,s,indent=2);s.write('\n')
