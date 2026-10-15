from pathlib import Path
import hashlib,json,sys
p,o=map(Path,sys.argv[1:3]);post=len(sys.argv)==4 and sys.argv[3]=='post'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb')as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
bind=json.loads((p/'source-link-bindings.json').read_text());actual={}
for n,v in bind.items():
 f=Path(v['host_path']);assert f.stat().st_size==v['bytes']and sha(f)==v['sha256'],n;actual[n]=v['sha256']
for n,v in json.loads((p/'harness-pins.json').read_text()).items():assert sha(p/n)==v,n
records={}
for l in(p/'build/sha256.txt').read_text().splitlines():h,n=l.split(maxsplit=1);records[n]=h
binary=sha(p/'build/q6_center_leaf_s85');assert records['q6_center_leaf_s85']==binary
for n in ('/src/q6_center_leaf.cpp','/src/private_q6_center.hpp','/common/mmvq-core.images.o','/frozen/libstrata_kernels.a','/frozen/libstrata_core.a'):assert records[n]==actual[n],n
assert json.loads((p/'build/image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
log=(p/'build/host-link.log').read_text();assert log.count('/usr/bin/ld')==1 and 'crtfastmath'not in log
for bad in ('ocloc','llvm-link','sycl-post-link','clang-offload-wrapper','llvm-foreach'):assert bad not in log
f=json.loads((p/'fixture-pins.json').read_text());model=Path(f['path']);assert model.stat().st_size==f['file_size']
fixtures={}
for n,v in f['saved_fixture_files'].items():
 x=Path(v['host_path']);assert x.stat().st_size==v['bytes']and sha(x)==v['sha256'],n;fixtures[n]=v['sha256']
# Future bounded reads only the two real slices; leaf also checks frozen S44 FNVs.
weights=[]
with model.open('rb')as s:
 for t in f['tensors']:
  s.seek(t['absolute_offset']);b=s.read(t['bytes']);assert len(b)==t['bytes'];weights.append(dict(t,sha256=hashlib.sha256(b).hexdigest()))
d={'status':'PASS_HOST_SOURCE_BUILD_FIXTURE_PINS','binary_sha256':binary,'inputs':actual,'saved_fixture_hashes':fixtures,'weight_slices':weights,'physical_card':0,'pci':'0000:0b:00.0','projection_only':True,'source_unchanged_common_AOT_reused':True}
if post:
 before=json.loads((o/'identity.json').read_text());assert before==d,'pre/post identity mismatch'
with(o/('post-identity.json'if post else'identity.json')).open('x',encoding='ascii')as s:json.dump(d,s,indent=2);s.write('\n')
