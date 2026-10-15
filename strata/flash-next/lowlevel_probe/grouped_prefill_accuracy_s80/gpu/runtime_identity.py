#!/usr/bin/env python3
"""Root-reviewed input/build preflight; no GPU or independent reference work."""
from pathlib import Path
import hashlib,json,sys
p,o,refs=map(Path,sys.argv[1:4]);post=len(sys.argv)==5 and sys.argv[4]=="post";fixtures=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-grouped-gemm-s42-prep/run-clean/exports')
def sha(f):
 h=hashlib.sha256()
 with f.open('rb')as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify(f,r):
 assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256'],str(f)
 return r
pins=json.loads((p/'harness-pins.json').read_text())
for name,digest in pins.items():assert sha(p/name)==digest,name
image='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
assert json.loads((p/'build/image.json').read_text())[0]['Id']==image
closure=json.loads((p/'root-build-closure.json').read_text())
assert closure['status']=='PASS_ROOT_BUILD_AUDIT'
exe=p/'build/grouped_prefill_leaf_s80'
assert sha(exe)==closure['binary_sha256']
assert sha(p/'grouped_prefill_leaf.cpp')==closure['source_sha256']
assert sha(p/'reference_bound.hpp')==closure['reference_bound_sha256']
records={}
for line in (p/'build/sha256.txt').read_text().splitlines():
 h,n=line.split(maxsplit=1);records[n]=h
assert records['grouped_prefill_leaf_s80']==closure['binary_sha256']
assert records['/src/grouped_prefill_leaf.cpp']==closure['source_sha256']
assert records['/src/reference_bound.hpp']==closure['reference_bound_sha256']
assert (p/'build/archives-before.sha256').read_bytes()==(p/'build/archives-after.sha256').read_bytes()
archive_pins={}
for line in (p/'build/archives-after.sha256').read_text().splitlines():
 h,n=line.split(maxsplit=1);assert n.startswith('/frozen/')
 f=p/'frozen-inputs'/n.removeprefix('/frozen/');assert sha(f)==h,str(f);archive_pins[n]=h
for line in(p/'expected-source-container.sha256').read_text().splitlines():
 h,n=line.split(maxsplit=1)
 f=p/n.removeprefix('/src/') if n.startswith('/src/') else Path('/mnt/vm_8tb/b70/build/strata-ab')/n.removeprefix('/work/')
 assert sha(f)==h,str(f)
old=json.loads((p/'fixture-pins.json').read_text())['files']
names=('weights.bin','fixtures.tsv','source.fp16','source_ids.i32','actual_gathered.fp16','reference_GU.f32','reference_H.fp16','reference_down.f32')
fixture_files={n:verify(fixtures/n,old[n]) for n in names}
r=json.loads((refs/'reference.json').read_text());assert r['status']=='PASS_CPU_FULL_GU_REFERENCE_GPU_IDENTITY_PENDING'
assert r['GPU_operand_identity_measured'] is False and r['ideal_H_down_reference'] is False
for n,rec in r['inputs'].items():assert n in old;verify(fixtures/n,rec);assert rec==old[n]
assert {'weights.bin','fixtures.tsv','actual_gathered.fp16'}<=set(r['inputs'])
for n,rec in r['files'].items():assert Path(n).name==n;verify(refs/n,rec)
needed={'gu-planes.fp16':104857600,'gu-T64.f64':17899520,'gu-bound.f64':17899520,'actual-gathered-changed.fp16':8949760}
for n,z in needed.items():assert r['files'][n]['bytes']==z,n
identity={'status':'PASS_INPUT_BUILD_REFERENCE_PROVENANCE_GPU_PREMISE_PENDING','binary_sha256':closure['binary_sha256'],
 'fixture_files':fixture_files,'reference_files':r['files'],'reference_json_sha256':sha(refs/'reference.json'),
 'root_build_closure_sha256':sha(p/'root-build-closure.json'),'archive_pins':archive_pins,'harness_pins':pins,'image':image,'physical_card':1,'pci':'0000:44:00.0',
 'timing_scope':'complete eager wave, canonical reset outside, DQ/GU/SiLU/down/dependency barriers and grouped metadata fill inside',
 'old_S42_strict_status':'FAILED_UNCHANGED','ideal_H_down_accuracy':False,'serving_qualified':False}
if post:
 assert identity==json.loads((o/'identity.json').read_text()),'pre/post identity mismatch'
 with(o/'post-input-pins.json').open('x',encoding='ascii')as f:json.dump({'status':'PASS_PRE_POST_INPUT_BUILD_REFERENCE_IDENTITY'},f);f.write('\n')
else:
 with(o/'identity.json').open('x',encoding='ascii')as f:json.dump(identity,f,indent=2);f.write('\n')
print(json.dumps({'status':identity['status'],'binary_sha256':closure['binary_sha256']}))
