from pathlib import Path
import struct,hashlib,json
p=Path(__file__).resolve().parent;b=bytearray((p/'iq3s-C6.zebin').read_bytes()); orig=bytes(b)
so=struct.unpack_from('<Q',b,40)[0]; se,sn,si=struct.unpack_from('<HHH',b,58); sh=[struct.unpack_from('<IIQQQQIIQQ',b,so+i*se) for i in range(sn)]; st=sh[si]; names=b[st[4]:st[4]+st[5]]
x=[s for s in sh if names[s[0]:].split(b'\0')[0].startswith(b'.text._ZTS')];assert len(x)==1;s=x[0];off=st[4]+s[0]; old=bytes(b[off:]).split(b'\0')[0];alias=b'.text.iq3s_c6';b[off:off+len(old)]=alias+b'\0'*(len(old)-len(alias));assert len(b)==len(orig)
assert b[s[4]:s[4]+s[5]]==orig[s[4]:s[4]+s[5]]
(p/'iq3s-C6-shortname.zebin').write_bytes(b)
(p/'alias-proof.json').write_text(json.dumps({'reason':'ocloc output filename exceeds filesystem NAME_MAX; metadata-only alias in copy','original_section':old.decode(),'original_section_name_bytes':len(old),'alias':alias.decode(),'changed_offsets':[i for i,(a,c) in enumerate(zip(orig,b)) if a!=c],'only_shstrtab_changes':all(st[4]<=i<st[4]+st[5] for i,(a,c) in enumerate(zip(orig,b)) if a!=c),'original_zebin_sha256':hashlib.sha256(orig).hexdigest(),'aliased_zebin_sha256':hashlib.sha256(b).hexdigest(),'main_text_offset':s[4],'main_text_bytes':s[5],'original_and_aliased_main_text_sha256':hashlib.sha256(orig[s[4]:s[4]+s[5]]).hexdigest()},indent=2)+'\n')
