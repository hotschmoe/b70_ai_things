#!/usr/bin/env python3
"""One preflight inside quiet CPU/card0 lease; does not touch GPU."""
from pathlib import Path
import hashlib,json,sys
p,o=map(Path,sys.argv[1:]);r=p.parent
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
pins={}
for line in(p/'expected-container.sha256').read_text().splitlines():
 h,n=line.split(maxsplit=1)
 if n.startswith('/src/'):f=p/n.removeprefix('/src/')
 elif n.startswith('/work/'):f=Path('/mnt/vm_8tb/b70/build/strata-ab')/n.removeprefix('/work/')
 elif n.startswith('/common/'):f=r/'iq4xs-rpw2-s59-prep/build'/n.removeprefix('/common/')
 elif n.startswith('/frozen/'):f=r/'iq4xs-rpw2-s59-prep/frozen-inputs'/n.removeprefix('/frozen/')
 elif n.startswith('/opt/intel/'):continue
 else:raise ValueError(n)
 assert sha(f)==h,str(f)
 pins[n]=h
records={}
for line in(p/'build/sha256.txt').read_text().splitlines():
 h,n=line.split(maxsplit=1);records[n]=h
assert records['iq3s_dpas_leaf_s75']==sha(p/'build/iq3s_dpas_leaf_s75')
for n in ['/src/iq3s_dpas_leaf.cpp','/src/private_iq3s_dpas_c6.hpp','/common/mmvq-core.images.o','/frozen/libstrata_kernels.a','/frozen/libstrata_core.a']:assert records[n]==pins[n]
assert json.loads((p/'build/image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
hostlog=(p/'build/host-link.log').read_text()
assert hostlog.count('/usr/bin/ld')==1
for forbidden in ['ocloc','llvm-link','sycl-post-link','clang-offload-wrapper','llvm-foreach']:assert forbidden not in hostlog
identity=json.loads((p/'weight-identity.json').read_text());model=Path(identity['path'])
assert model.stat().st_size==identity['file_size']
with model.open('rb') as f:
 for t in identity['tensors']:
  f.seek(t['absolute_offset']);b=f.read(t['bytes']);assert len(b)==t['bytes']
  assert hashlib.sha256(b).hexdigest()==t['sha256'],t['tensor']
identity.update(binary_sha256=records['iq3s_dpas_leaf_s75'],immutable_link_inputs={k:records[k] for k in records if k.startswith(('/frozen/','/common/'))},source_pins=pins,
 timing_scope='same-process projection-only graph20; canonical Q8 outside graphs; 8 OCCO/COOC rounds per shape; no significance or serving claim',
 image='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2',physical_card=0,pci='0000:0b:00.0',external_profiler='none')
with(o/'identity.json').open('x',encoding='ascii')as f:json.dump(identity,f,indent=2);f.write('\n')
