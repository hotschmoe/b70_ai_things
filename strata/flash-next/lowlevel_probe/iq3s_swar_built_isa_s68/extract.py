#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,struct,time
p=Path(__file__).parent;r=p.parent;b=r/'iq3s-dense-oracle-s65-prep/build';exe=b/'iq3s_dense_leaf_s65'
sha=lambda b:hashlib.sha256(b).hexdigest()
expected='4a81f24dcbb2dee280cfe009035e58f8c605a8f47cf970c5d080e8242ca30902'
records={}
for l in(b/'sha256.txt').read_text().splitlines():h,n=l.split(maxsplit=1);records[n]=h
assert records['iq3s_dense_leaf_s65']==expected
st=exe.stat();readbytes=0;t0=time.monotonic()
with exe.open('rb') as f:
 def rd(off,n):
  global readbytes
  f.seek(off);data=f.read(n);assert len(data)==n;readbytes+=n;return data
 def elf(base):
  h=rd(base,64);assert h[:6]==b'\x7fELF\x02\x01'
  so=struct.unpack_from('<Q',h,40)[0];se,sn,si=struct.unpack_from('<HHH',h,58)
  raw=rd(base+so,se*sn);ss=[struct.unpack_from('<IIQQQQIIQQ',raw,i*se) for i in range(sn)]
  ns=ss[si];names=rd(base+ns[4],ns[5]);sections={names[s[0]:].split(b'\0')[0].decode():s for s in ss}
  size=max(so+se*sn,max(s[4]+s[5] for s in ss if s[1]!=8))
  return sections,size
 top,_=elf(0);bundle=top['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'];off=bundle[4];end=off+bundle[5];index=0;selected={}
 while off<end:
  ss,size=elf(off);z=ss.get('.ze_info');info=rd(off+z[4],z[5]).decode()if z else ''
  arm=None
  if 'PrivateC6ILb0EE' in info:arm='original'
  elif 'PrivateC6ILb1EE' in info:arm='candidate'
  elif 'launch_mmvq_multi_ncILi21ELb0EE' in info and 'dpct_kernel_scalarILi21EESK_ILi6EESK_ILi0EESK_ILi1EEEE' in info:arm='public'
  if arm:
   assert arm not in selected,(arm,index)
   data=rd(off,size)
   needle='PrivateC6ILb0EE' if arm=='original' else 'PrivateC6ILb1EE' if arm=='candidate' else 'mmvq_multi_kernel_3404b2'
   kernel_sections={n:s for n,s in ss.items()if n.startswith('.text.') and needle in n};assert len(kernel_sections)==1,kernel_sections.keys()
   name,s=next(iter(kernel_sections.items()));text=data[s[4]:s[4]+s[5]]
   op=ss.get('.misc.buildOptions');options=rd(off+op[4],op[5]).decode().strip('\0')if op else None
   (p/(arm+'.zebin')).write_bytes(data);(p/(arm+'.ze_info')).write_text(info)
   selected[arm]={'index':index,'offset':off,'zebin_bytes':size,'zebin_sha256':sha(data),'kernel':name.removeprefix('.text.'),
    'text_section':name,'text_bytes':len(text),'text_sha256':sha(text),'build_options':options,'resource_metadata':info}
  off=(off+size+15)//16*16;index+=1
 assert set(selected)=={'public','original','candidate'},selected.keys()
 assert selected['public']['zebin_sha256']=='bad2ccded9f4136e6809825844815652eba954f059b807774c45d7916d92beb4'
assert exe.stat().st_size==st.st_size and exe.stat().st_mtime_ns==st.st_mtime_ns
report={'exe':str(exe),'exe_sha256_from_actual_build_record':expected,'exe_stat':{'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino},'exe_stat_unchanged':True,
 'full_exe_hash_not_repeated':True,'build_records_sha256':sha((b/'sha256.txt').read_bytes()),'images_indexed':index,'metadata_and_selected_bytes_read':readbytes,
 'seconds':time.monotonic()-t0,'public_canonical_image_matches_S66':True,'selected':selected,'new_builds':0,'GPU_touches':0}
(p/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'EXTRACTED_THREE_EXISTING_IMAGES_PUBLIC_CANONICAL_MATCH','images':index,'read_bytes':readbytes,'seconds':report['seconds'],'selected':{k:{n:v[n]for n in ['index','zebin_bytes','text_bytes','zebin_sha256','text_sha256']}for k,v in selected.items()}},indent=2))
