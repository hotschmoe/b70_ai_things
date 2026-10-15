#!/usr/bin/env python3
"""Source/binary/public object/root build closure; CPU-only, no GPU identity claim."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent;S=P.parent/'attn-value-binding-s92-prep';B=S/'build';OUT=Path(sys.argv[1]);mode=sys.argv[2]
def sha(f):
 with Path(f).open('rb')as x:return hashlib.file_digest(x,'sha256').hexdigest()
pins={}
for row in(S/'expected-source.sha256').read_text().splitlines():
 h,n=row.split(maxsplit=1)
 if n.startswith('/src/'):f=S/n[5:]
 elif n.startswith('/work/'):f=Path('/mnt/vm_8tb/b70/build/strata-ab')/n[6:]
 else:raise ValueError(n)
 assert sha(f)==h,n;pins[n]=h
for row in(P/'expected-runner.sha256').read_text().splitlines():
 h,n=row.split(maxsplit=1);assert sha(P/n)==h,n
closure=json.loads((B/'root-build-closure.json').read_text());assert closure['status']=='PASS_ROOT_BUILD_AUDIT'
assert closure['binary_sha256']==sha(B/'attn_value_probe')
assert closure['expected_source_sha256']==sha(S/'expected-source.sha256')
assert closure['public_object_sha256']==pins['/src/frozen/qsa_decode_attn.dp.cpp.o']=='bbae92dcf61427f35078d498a240b1c9f23862e8bf13f7b63d95252e3b06baf2'
assert closure['kernel_bindings_audited']is True and closure['host_metadata_bindings_audited']is True and closure['offline_only']is True and closure['host_link_AOT_actions']==0
for row in(B/'sha256.txt').read_text().splitlines():
 h,n=row.split(maxsplit=1);assert sha(S/n[5:]if n.startswith('/src/')else B/n)==h,n
assert json.loads((B/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
log=(B/'host-link.log').read_text()
for word in ['ocloc','sycl-post-link','llvm-foreach','clang-offload-wrapper','crtfastmath.o']:assert word not in log,word
actual={'binary_sha256':closure['binary_sha256'],'root_build_closure_sha256':sha(B/'root-build-closure.json'),'source_pins':pins,'runner_pins_sha256':sha(P/'expected-runner.sha256'),'execution':'one process/eager/no graphs/no models','requested_physical_card':0,'requested_PCI':'0000:0b:00.0','ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:*','LANECELL':'absent','GPU_identity_scope':'Requested binding only; leaf currently prints SYCL name. No independent PCI read asserted by this CPU check.'}
if mode=='pre':name='identity.json'
else:
 assert mode=='post'and json.loads((OUT/'identity.json').read_text())==actual;name='post-identity.json'
with(OUT/name).open('x',encoding='ascii')as f:json.dump(actual,f,indent=2);f.write('\n');f.flush()
