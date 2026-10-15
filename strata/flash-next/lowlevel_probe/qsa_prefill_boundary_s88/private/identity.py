#!/usr/bin/env python3
"""Explicit source/binary/root-build closure, inside caller's leased run."""
from pathlib import Path
import hashlib,json,sys
p,out=map(Path,sys.argv[1:3]);mode=sys.argv[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins={}
for row in(p/'expected-source.sha256').read_text().splitlines():
 h,n=row.split(maxsplit=1)
 if n.startswith('/src/'):f=p/n[5:]
 elif n.startswith('/work/'):f=Path('/mnt/vm_8tb/b70/build/strata-ab')/n[6:]
 elif n.startswith('/public/'):f=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/qsa-prefill-tiled-s84-prep-v3/build')/n[8:]
 else:raise ValueError(n)
 assert sha(f)==h,n
 pins[n]=h
b=p/'build';v=json.loads((b/'root-build-closure.json').read_text())
assert v['status']=='PASS_ROOT_BUILD_AUDIT'
assert v['binary_sha256']==sha(b/'scorer_leaf_s88')
assert v['expected_source_sha256']==sha(p/'expected-source.sha256')
assert v['public_object_sha256']==pins['/src/frozen/qsa_select.dp.cpp.o']
assert v['kernel_bindings_audited'] is True and v['offline_only'] is True and v['host_link_AOT_actions']==0
assert json.loads((b/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
log=(b/'host-link.log').read_text()
for term in ['ocloc','sycl-post-link','llvm-foreach','clang-offload-wrapper','crtfastmath.o']:assert term not in log,term
for row in(b/'sha256.txt').read_text().splitlines():
 h,n=row.split(maxsplit=1);f=p/n[5:] if n.startswith('/src/') else Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/qsa-prefill-tiled-s84-prep-v3/build')/n[8:] if n.startswith('/public/') else b/n
 assert sha(f)==h,n
actual={'source_pins':pins,'binary_sha256':v['binary_sha256'],'root_build_closure_sha256':sha(b/'root-build-closure.json'),'execution':'one_process_eager_only','card':0,'pci':'0000:0b:00.0','retries':0}
if mode=='pre':
 with(out/'identity.json').open('x',encoding='ascii')as f:json.dump(actual,f,indent=2);f.write('\n')
else:
 assert mode=='post' and json.loads((out/'identity.json').read_text())==actual
 with(out/'post-identity.json').open('x',encoding='ascii')as f:json.dump(actual,f,indent=2);f.write('\n')
