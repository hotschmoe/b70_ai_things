from pathlib import Path
import hashlib,json,re
P=Path(__file__).resolve().parent;R=P.parent;A=R/'routed-iq2-multi-s98-build-audit-v2';O=R/'routed-iq2-multi-s98-candidate-isa'
sha=lambda b:hashlib.sha256(b).hexdigest()
f=A/'facts.json';v=json.loads(f.read_text());assert v['binary_sha256']=='f749a06942ca078afdb69254ac6f035f03c5789602d8c92f8790907c0322ad94'
assert v['status']=='PASS_THREE_DISTINCT_HOST_BINDINGS_QUALIFIED_PUBLIC_COMPUTATIONAL_IDENTITY_ROOT_RESOURCE_REVIEW_PENDING'
s=v['selected']['candidate'];assert s['kernel']=='_ZTSN3s9811PrivateGu17ILb1EEE'
assert s['image_sha256']=='94b7bfa94926bc987e7f86eb1d79b17158a48912db7d1f7d74a7d300cf960c8e'
b=(A/'candidate.zebin').read_bytes();assert sha(b)==s['image_sha256'];assert s['all_kernel_names_in_this_selected_image']==[s['kernel'],'Intel_Symbol_Table_Void_Program']
assert len('.text.'+s['kernel']+'.asm')<255
O.mkdir(exist_ok=False);(O/'tmp').mkdir();(O/'disasm').mkdir();(O/'candidate-input.zebin').write_bytes(b)
(O/'input.json').write_text(json.dumps({'facts_sha256':sha(f.read_bytes()),'binary_sha256':v['binary_sha256'],'selected':s,'alias':'none needed: selected short name is below NAME_MAX; exact image COPY only','no_public_decode':True,'no_all_image_walk':True},indent=2)+'\n',encoding='ascii')
