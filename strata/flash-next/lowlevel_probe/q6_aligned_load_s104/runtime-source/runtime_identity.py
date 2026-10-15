from pathlib import Path
import hashlib,json,sys
p,o=map(Path,sys.argv[1:3]);post=len(sys.argv)==4 and sys.argv[3]=='post'
R=p.parent;S=R/'q6-a4-private-prep';B=S/'build'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb')as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
bind=json.loads((S/'source-link-bindings.json').read_text());actual={}
for n,v in bind.items():
 f=Path(v['host_path']);assert f.stat().st_size==v['bytes']and sha(f)==v['sha256'],n;actual[n]=v['sha256']
for n,v in json.loads((p/'harness-pins.json').read_text()).items():assert sha(p/n)==v,n
records={}
for l in(B/'sha256.txt').read_text().splitlines():h,n=l.split(maxsplit=1);records[n]=h
binary=sha(B/'q6_a4_leaf_s104');assert records['q6_a4_leaf_s104']==binary=='632f9f86ec2bea2b030286da62c75aa3581c01190a2ecb203f6b1f6f4c7e8e8c'
for n in ('/src/q6_a4_leaf.cpp','/src/private_q6_a4.hpp','/common/mmvq-core.images.o','/frozen/libstrata_kernels.a','/frozen/libstrata_core.a'):assert records[n]==actual[n],n
assert json.loads((B/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
log=(B/'host-link.log').read_text();assert log.count('/usr/bin/ld')==1 and 'crtfastmath'not in log
for bad in ('ocloc','llvm-link','sycl-post-link','clang-offload-wrapper','llvm-foreach'):assert bad not in log
f=json.loads((p/'fixture-pins.json').read_text());model=Path(f['path']);assert model.stat().st_size==f['file_size']
fixtures={}
for n,v in f['saved_fixture_files'].items():
 x=Path(v['host_path']);assert x.stat().st_size==v['bytes']and sha(x)==v['sha256'],n;fixtures[n]=v['sha256']
# No model payload reread here: leaf checks exact two slices' S44 FNVs.
st=model.stat();weights={'path':str(model),'device':st.st_dev,'inode':st.st_ino,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'two_slice_identities':f['tensors']}
proof=R/'q6-a4-s104-compiled-audit/facts.json';audit=json.loads(proof.read_text());assert audit['status']=='PASS_THREE_HOST_BINDINGS_ORIGINAL_S85_COMPUTATIONAL_REUSE_CANDIDATE_ISA_PENDING_ROOT_GO'and audit['binary_sha256']==binary
assert audit['original_comparison']['computational_prefix_exact']
# Root full-read/model-review/explicit execution GO is the external launch prerequisite, as in S85.
d={'status':'PASS_HOST_SOURCE_BUILD_FIXTURE_PINS','binary_sha256':binary,'inputs':actual,'saved_fixture_hashes':fixtures,'weight_slices':weights,'physical_card':0,'pci':'0000:0b:00.0','projection_only':True,'source_unchanged_common_AOT_reused':True,'compiled_audit_sha256':sha(proof)}
if post:
 before=json.loads((o/'identity.json').read_text());assert before==d,'pre/post identity mismatch'
with(o/('post-identity.json'if post else'identity.json')).open('x',encoding='ascii')as s:json.dump(d,s,indent=2);s.write('\n')
