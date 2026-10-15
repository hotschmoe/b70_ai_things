from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent;R=P.parent
mode,out=sys.argv[1],Path(sys.argv[2]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins=json.loads((P/'harness-pins.json').read_text());actual={n:sha(Path(n)) for n in pins};assert actual==pins
binary=P/'build/hc_more_oracle';binary_sha=sha(binary)
build=json.loads((P/'root-build-closure.json').read_text())
assert build['binary_sha256']==binary_sha and build['adapter_source_sha256']==sha(P/'hc_more_oracle.cpp')
assert build['frozen_kernel_archive_sha256']=='f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8'
assert build['common_wrapper_sha256']=='8f5a7efea1c815b497edfaa1bd33e02d86d4cf9ceea38fd9190d44bacfd7cd21'
assert build['HC_member_sha256']=='27d0ab6aff30126bd317584b4b2cbacd0b4667764f9cf624738625c787c07d99'
assert build['verify_support_member_sha256']=='584e694fc1ece4740fe4550e2c34fafd28a77921d81960039d01f3c27dbb6613'
x={'mode':mode,'pins':actual,'binary_sha256':binary_sha,'root_build_closure_sha256':sha(P/'root-build-closure.json'),'GPU_work':False}
if mode=='post':
 pre=json.loads((out/'host-pre.json').read_text());assert pre['pins']==actual and pre['binary_sha256']==binary_sha and pre['root_build_closure_sha256']==x['root_build_closure_sha256']
 result=json.loads((out/'comparison.json').read_text());assert result['status']=='PASS_ALL_FOUR_PROCESS_NUMERICAL_CAPTURE_GATES'
 assert all((out/label/'oracle.exit').read_text().strip()=='0' and (out/label/'container-removed').read_text().strip()=='1' for label in ['a1','b1','b2','a2'])
 x.update(comparison_sha256=sha(out/'comparison.json'),binary_files=4*69,DOT_files=4*15,whole_state_bytes_each=747776,scope='All artifact SHA values and full-byte comparisons retained in comparison.json; no duplicate file copy/hash campaign')
else:assert mode=='pre'
x['status']='PASS_HOST_BINDING_ONLY'
with (out/f'host-{mode}.json').open('x',encoding='ascii') as f:json.dump(x,f,indent=2);f.write('\n')
print(json.dumps({'status':x['status'],'mode':mode}))
