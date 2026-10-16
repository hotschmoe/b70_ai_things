"""Optional selected-image section-name alias only, after explicit root GO."""
from pathlib import Path
import hashlib,json,struct,sys
from compare import sections,cstring
O=Path(sys.argv[1]);image=(O/'selected.zebin').read_bytes();j=json.loads((O/'selection.json').read_text())
assert hashlib.sha256(image).hexdigest()==j['selected']['image_sha256']
rows,names,ss=sections(image);name=j['selected']['kernel'];t=ss['.text.'+name][0]
si=struct.unpack_from('<H',image,62)[0];ns=rows[si];offset=ns[4]+t[0]
old=cstring(image,offset);new=b'.text.gu17_selected';assert old==('.text.'+name).encode()and len(new)<len(old)
alias=bytearray(image);alias[offset:offset+len(old)]=new+b'\0'*(len(old)-len(new))
assert alias[t[4]:t[4]+t[5]]==image[t[4]:t[4]+t[5]]
assert all(ns[4]<=i<ns[4]+ns[5]for i,(a,b)in enumerate(zip(image,alias))if a!=b)
(O/'selected-short.zebin').write_bytes(alias)
(O/'alias-proof.json').write_text(json.dumps({'status':'ONLY_SELECTED_SECTION_NAME_TABLE_COPY_CHANGED','original_sha256':hashlib.sha256(image).hexdigest(),'alias_sha256':hashlib.sha256(alias).hexdigest(),'text_exact':True,'selected_images':1},indent=2)+'\n',encoding='ascii')
